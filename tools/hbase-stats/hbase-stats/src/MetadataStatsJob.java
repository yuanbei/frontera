import org.apache.commons.codec.DecoderException;
import org.apache.commons.codec.binary.Hex;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableMapReduceUtil;
import org.apache.hadoop.hbase.mapreduce.TableMapper;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

public class MetadataStatsJob extends Configured implements Tool {
    public static class MetadataStatsMapper extends TableMapper<Text, IntWritable> {
        private HashMap<String, Integer> hostStats = new HashMap<String, Integer>();

        public void map(ImmutableBytesWritable row, Result value, Context context) throws InterruptedException, IOException {
            String rawUrl = null;
            byte state = -1;
            for (Cell c: value.rawCells()) {
                String qualifier = new String(c.getQualifierArray(), c.getQualifierOffset(), c.getQualifierLength());
                if (qualifier.equals("url"))
                    rawUrl = new String(c.getValueArray(), c.getValueOffset(), c.getValueLength());

                if (qualifier.equals("state")) {
                    ByteBuffer bb = ByteBuffer.wrap(c.getValueArray(), c.getValueOffset(), c.getValueLength());
                    bb.order(ByteOrder.BIG_ENDIAN);
                    state = bb.get();
                }
            }

            if (state > 1 && rawUrl != null) {
                try {
                    URL url = new URL(rawUrl);
                    int count = 0;
                    if (hostStats.containsKey(url.getHost()))
                        count = hostStats.get(url.getHost());
                    hostStats.put(url.getHost(), count + 1);
                } catch (MalformedURLException murl) {
                    murl.printStackTrace(System.err);
                }
            }
        }

        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (Map.Entry<String, Integer> entry: hostStats.entrySet())
                context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
        }
    }

    public static class MetadataStatsSummingReducer extends Reducer<Text, IntWritable, Text, Text> {
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            long sum = 0;
            for (IntWritable value: values)
                sum += value.get();
            context.write(key, new Text(String.format("%d", sum)));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration config = HBaseConfiguration.create();
        Tool tool = new MetadataStatsJob();
        System.exit(ToolRunner.run(config, tool, args));
    }

    public int run(String[] args) throws Exception {
        Configuration config = getConf();
        Job job = Job.getInstance(config, "Frontera metadata stats");
        job.setJarByClass(MetadataStatsJob.class);
        Scan scan = new Scan();
        scan.setCaching(500);
        scan.setCacheBlocks(false);

        String tableName = String.format("%s:metadata", args[0]);
        TableMapReduceUtil.initTableMapperJob(
                tableName,
                scan,
                MetadataStatsMapper.class,
                Text.class,
                IntWritable.class,
                job);

        job.setReducerClass(MetadataStatsSummingReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        Path outputPath = new Path(args[1]);
        TextOutputFormat.setOutputPath(job, outputPath);
        return job.waitForCompletion(true) ? 0 : -1;
    }
}
