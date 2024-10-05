import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.log4j.Logger

import scala.jdk.CollectionConverters.*

object SlidingWindowJob {
  private val logger = Logger.getLogger(getClass)

  private class SlidingWindowMapper extends Mapper[LongWritable, Text, Text, Text] {
    private val windowSize = 4 // Adjust as needed

    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      val inputLine = value.toString
      logger.info(s"Processing input line: $inputLine")

      val parts = inputLine.split("\t")
      if (parts.nonEmpty) {
        val token = parts(1)
        logger.info(s"Extracted token: $token")
        context.write(new Text("data"), new Text(token))
      } else {
        logger.warn(s"Skipping empty input line")
      }
    }
  }

  private class SlidingWindowReducer extends Reducer[Text, Text, Text, Text] {
    private val windowSize = 4 // Adjust as needed

    override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      val tokens = values.asScala.map(_.toString).toArray
      logger.info(s"Reducer received ${tokens.length} tokens")

      tokens.sliding(windowSize + 1).zipWithIndex.foreach { case (window, idx) =>
        if (window.length == windowSize + 1) {
          val (inputWindow, label) = window.splitAt(windowSize)
          val inputStr = inputWindow.mkString(",")
          val labelStr = label.head
          logger.info(s"Emitting window $idx: input=$inputStr, label=$labelStr")
          context.write(new Text(s"input_$idx"), new Text(inputStr))
          context.write(new Text(s"label_$idx"), new Text(labelStr))
        } else {
          logger.warn(s"Skipping window $idx due to insufficient length: ${window.mkString(",")}")
        }
      }
    }
  }

  def runJob(conf: Configuration, input: Path, output: Path): Unit = {
    val job = Job.getInstance(conf, "Sliding Window for Embedding")
    job.setJarByClass(this.getClass)
    job.setMapperClass(classOf[SlidingWindowMapper])
    job.setReducerClass(classOf[SlidingWindowReducer])
    job.setMapOutputKeyClass(classOf[Text])
    job.setMapOutputValueClass(classOf[Text])
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])
    FileInputFormat.addInputPath(job, input)
    FileOutputFormat.setOutputPath(job, output)

    //  Delete output path if it exists
    val fs = FileSystem.get(conf)
    if (fs.exists(output)) {
      fs.delete(output, true)
    }

    logger.info(s"Starting Sliding Window job with input: $input and output: $output")
    if (!job.waitForCompletion(true)) {
      logger.error("Sliding Window job failed")
      System.exit(1)
    }
    logger.info("Sliding Window job completed successfully")
  }
}