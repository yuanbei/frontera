import com.google.protobuf.ByteString;
import com.scrapinghub.frontera.hbasestats.QueueRecordOuter;
import org.apache.commons.codec.binary.Hex;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.filter.Filter;
import org.apache.hadoop.hbase.filter.PrefixFilter;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableMapReduceUtil;
import org.apache.hadoop.hbase.mapreduce.TableMapper;
import org.apache.hadoop.hbase.mapreduce.TableReducer;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.msgpack.MessagePack;
import org.msgpack.annotation.Message;
import org.msgpack.packer.Packer;
import org.msgpack.type.ArrayValue;
import org.msgpack.type.Value;
import org.msgpack.unpacker.Converter;
import org.msgpack.unpacker.Unpacker;
import org.msgpack.unpacker.UnpackerIterator;
import sun.misc.*;

import java.io.*;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.List;
import java.util.Map;
import java.util.Random;


public class ShuffleQueueJob extends Configured implements Tool {
    @Message
    public static class QueueItem {
        public byte[] fingerprint;
        public int hostCrc32;
        public String url;
        public float score;
    }

    public static class HostsDumpMapper extends TableMapper<IntWritable, BytesWritable> {
        public enum Counters {FINGERPRINTS_COUNT}
        private MessagePack msgPack = new MessagePack();

        public void map(ImmutableBytesWritable row, Result value, Context context) throws InterruptedException, IOException {
            String rk = new String(row.get(), row.getOffset(), row.getLength(), "US-ASCII");
            String[] rkParts = rk.split("_");
            QueueRecordOuter.QueueRecord.Builder builder = QueueRecordOuter.QueueRecord.newBuilder();
            builder.setPartitionId(Integer.parseInt(rkParts[0]));
            builder.setTimestamp(Long.parseLong(rkParts[3]));

            for (Cell c: value.rawCells()) {
                String column = new String(c.getQualifierArray(), c.getQualifierOffset(), c.getQualifierLength(), "US-ASCII");
                String[] intervals = column.split("_");
                builder.setStartInterval(Float.parseFloat(intervals[0]));
                builder.setEndInterval(Float.parseFloat(intervals[1]));

                if (c.getValueLength() == 0)
                    continue;
                ByteArrayInputStream bis = new ByteArrayInputStream(c.getValueArray(), c.getValueOffset(), c.getValueLength());
                Unpacker unpacker = msgPack.createUnpacker(bis);
                for (Value raw: unpacker) {
                    QueueItem item = new QueueItem();
                    if (!raw.isArrayValue())
                        continue;
                    ArrayValue array = raw.asArrayValue();
                    item.fingerprint = array.get(0).asRawValue().getByteArray();
                    item.hostCrc32 = array.get(1).asIntegerValue().getInt();
                    item.url = array.get(2).asRawValue().getString();
                    item.score = array.get(3).asFloatValue().getFloat();

                    builder.setFingerprint(ByteString.copyFrom(item.fingerprint));
                    builder.setHostCrc32(item.hostCrc32);
                    builder.setUrl(item.url);
                    builder.setScore(item.score);

                    IntWritable key = new IntWritable(item.hostCrc32);
                    BytesWritable val = new BytesWritable(builder.build().toByteArray());
                    context.write(key, val);
                    context.getCounter(Counters.FINGERPRINTS_COUNT).increment(1);
                }
            }
        }
    }

    public static class BuildQueueReducerCommon extends Reducer<BytesWritable, BytesWritable, Text, Text> {
        public class BuildQueueCommon extends BuildQueue {
            public void flushBuffer(Reducer.Context context, long timestamp) throws IOException, InterruptedException {
                for (Map.Entry<String, List<QueueRecordOuter.QueueRecord>> entry : buffer.entrySet()) {
                    List<QueueRecordOuter.QueueRecord> recordList = entry.getValue();
                    int count = recordList.size();
                    String rk = String.format("%s_%d", entry.getKey(), timestamp);
                    StringBuffer sBuffer = new StringBuffer();
                    sBuffer.append(count);
                    byte[] fingerprint = new byte[20];
                    for (QueueRecordOuter.QueueRecord record : entry.getValue()) {
                        record.getFingerprint().copyTo(fingerprint, 0);
                        sBuffer.append("\t");
                        sBuffer.append(Hex.encodeHexString(fingerprint) + "," + record.getHostCrc32());
                    }
                    context.write(new Text(rk), new Text(sBuffer.toString()));
                }
                buffer.clear();
            }
        }

        final BuildQueueCommon buildQueue = new BuildQueueCommon();

        public void setup(Context context) throws IOException, InterruptedException {
            buildQueue.setup(context);
        }

        public void reduce(BytesWritable hostCrc32, Iterable<BytesWritable> values, Context context) throws IOException, InterruptedException {
            buildQueue.reduce(values, context);
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            buildQueue.cleanup(context);
        }
    }

    public static class BuildQueueReducerHbase extends TableReducer<IntWritable, BytesWritable, ImmutableBytesWritable> {
        public enum Counters {ITEMS_PRODUCED, ROWS_PUT}

        public class BuildQueueHbaseMsgPack extends BuildQueue {
            private final byte[] CF = "f".getBytes();
            private byte[] salt = new byte[4];
            private MessagePack messagePack = new MessagePack();

            public void flushBuffer(Reducer.Context context, long timestamp) throws IOException, InterruptedException {
                for (Map.Entry<String, List<QueueRecordOuter.QueueRecord>> entry : buffer.entrySet()) {
                    List<QueueRecordOuter.QueueRecord> recordList = entry.getValue();
                    RND.nextBytes(salt);
                    String saltStr = new String(Hex.encodeHex(salt, true));
                    String rk = String.format("%s_%d_%s", entry.getKey(), timestamp, saltStr);

                    ByteArrayOutputStream bos = new ByteArrayOutputStream();
                    Packer packer = messagePack.createPacker(bos);
                    String column = null;
                    QueueItem item = new QueueItem();
                    for (QueueRecordOuter.QueueRecord record : recordList) {
                        if (column == null)
                            column = String.format("%.3f_%.3f", record.getStartInterval(), record.getEndInterval());
                        item.fingerprint = record.getFingerprint().toByteArray();
                        item.hostCrc32 = record.getHostCrc32();
                        item.url = record.getUrl();
                        item.score = record.getScore();

                        packer.write(item);
                        context.getCounter(Counters.ITEMS_PRODUCED).increment(1);
                    }

                    Put put = new Put(Bytes.toBytes(rk));
                    put.add(CF, column.getBytes(), bos.toByteArray());
                    context.write(null, put);
                    context.getCounter(Counters.ROWS_PUT).increment(1);
                }
                buffer.clear();
            }
        }

        final BuildQueueHbaseMsgPack buildQueue = new BuildQueueHbaseMsgPack();
        public void setup(Context context) throws IOException, InterruptedException {
            buildQueue.setup(context);
            buildQueue.setPerHostLimit(100);
        }

        public void reduce(IntWritable hostCrc32, Iterable<BytesWritable> values, Context context) throws IOException, InterruptedException {
            buildQueue.reduce(values, context);
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            buildQueue.cleanup(context);
        }
    }

    public int run(String [] args) throws Exception {
        Configuration config = getConf();
        config.set("frontera.hbase.namespace", args[0]);

        boolean rJob = runBuildQueueFromSequenceFile(args);
        //boolean rJob = runDumpQueueToSequenceFile(args);
        if (!rJob)
            throw new Exception("Error during queue building.");
        return 0;
    }

    private boolean runBuildQueueHbase(String[] args) throws Exception {
        Configuration config = getConf();
        Job job = Job.getInstance(config, "BuildQueue Job");
        job.setJarByClass(ShuffleQueueJob.class);
        Scan scan = new Scan();
        scan.setCaching(500);
        scan.setCacheBlocks(false);

        String sourceTable = String.format("%s:%s", config.get("frontera.hbase.namespace"), args[1]);
        String targetTable = String.format("%s:%s", config.get("frontera.hbase.namespace"), args[2]);
        TableMapReduceUtil.initTableMapperJob(
                sourceTable,
                scan,
                HostsDumpMapper.class,
                IntWritable.class,
                BytesWritable.class,
                job);

        TableMapReduceUtil.initTableReducerJob(
                targetTable,
                BuildQueueReducerHbase.class,
                job);
        return job.waitForCompletion(true);
    }

    private boolean runDumpQueueToSequenceFile(String[] args) throws Exception {
            Configuration config = getConf();
            Job job = Job.getInstance(config, "Dumping queue to sequence file.");
            job.setJarByClass(ShuffleQueueJob.class);
            Scan scan = new Scan();
            scan.setCaching(500);
            scan.setCacheBlocks(false);
            String tableName = String.format("%s:%s", config.get("frontera.hbase.namespace"), args[1]);
            TableMapReduceUtil.initTableMapperJob(
                    tableName,
                    scan,
                    HostsDumpMapper.class,
                    IntWritable.class,
                    BytesWritable.class,
                    job);

            job.setOutputFormatClass(SequenceFileOutputFormat.class);
            SequenceFileOutputFormat.setOutputPath(job, new Path(args[2]));
            SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(BytesWritable.class);
            job.setNumReduceTasks(0);
            return job.waitForCompletion(true);
        }

    private boolean runBuildQueueFromSequenceFile(String[] args) throws Exception {
        Configuration config = getConf();
        Job job = Job.getInstance(config, "BuildQueue Job");
        job.setJarByClass(ShuffleQueueJob.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(BytesWritable.class);
        job.setInputFormatClass(SequenceFileInputFormat.class);
        SequenceFileInputFormat.addInputPath(job, new Path(args[1]));

        String targetTable = String.format("%s:%s", config.get("frontera.hbase.namespace"), args[2]);
        TableMapReduceUtil.initTableReducerJob(
                targetTable,
                BuildQueueReducerHbase.class,
                job);

        return job.waitForCompletion(true);
    }

    private boolean runBuildQueueDebug(String[] args) throws Exception {
        Configuration config = getConf();
        Job job = Job.getInstance(config, "BuildQueue Job (debug)");
        job.setJarByClass(ShuffleQueueJob.class);
        Scan scan = new Scan();
        scan.setCaching(500);
        scan.setCacheBlocks(false);

        String tableName = String.format("%s:queue", config.get("frontera.hbase.namespace"));
        TableMapReduceUtil.initTableMapperJob(
                tableName,
                scan,
                HostsDumpMapper.class,
                IntWritable.class,
                BytesWritable.class,
                job);

        job.setReducerClass(BuildQueueReducerCommon.class);
        Path outputPath = new Path(String.format("%s/hostsize", args[1]));
        FileOutputFormat.setOutputPath(job, outputPath);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(1);
        return job.waitForCompletion(true);
    }

    public static void main(String[] args) throws Exception {
        Configuration config = HBaseConfiguration.create();
        Tool tool = new ShuffleQueueJob();
        System.exit(ToolRunner.run(config, tool, args));
    }
}