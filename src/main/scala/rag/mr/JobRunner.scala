package rag.mr

import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat, NLineInputFormat}
import org.apache.hadoop.mapreduce.lib.output.{FileOutputFormat, TextOutputFormat}
import org.apache.hadoop.io.{IntWritable, Text}
import org.apache.hadoop.fs.Path
import org.apache.hadoop.conf.Configuration
import rag.Cfg
import scala.util.Try

object JobRunner {
  def main(args: Array[String]): Unit = {
    val cfg = Cfg.load()
    // Allow overriding from command line
    val inputList  = if (args.length > 0) args(0) else cfg.inputList
    val outStatus  = if (args.length > 1) args(1) else (cfg.outIndex + "/_mr-status")
    val reducers   = sys.env.get("REDUCERS").flatMap(s => Try(s.toInt).toOption).getOrElse(2)

    val conf = new Configuration()
    // Local mode so you can run WITHOUT Hadoop daemons:
    conf.set("mapreduce.framework.name", "local")
    conf.set("fs.defaultFS", "file:///")

    val job  = Job.getInstance(conf, "RAG Lucene Indexer (Local)")
    job.setJarByClass(classOf[RagMapper])

    job.setMapperClass(classOf[RagMapper])
    job.setReducerClass(classOf[RagShardReducer])
    job.setMapOutputKeyClass(classOf[IntWritable])
    job.setMapOutputValueClass(classOf[Text])
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])
    job.setOutputFormatClass(classOf[TextOutputFormat[Text,Text]])

    job.setInputFormatClass(classOf[NLineInputFormat])
    NLineInputFormat.setNumLinesPerSplit(job, 1)
    FileInputFormat.addInputPath(job, new Path(inputList))
    FileOutputFormat.setOutputPath(job, new Path(outStatus))
    job.setNumReduceTasks(reducers)

    val ok = job.waitForCompletion(true)
    if (!ok) sys.error("MR failed")
  }
}