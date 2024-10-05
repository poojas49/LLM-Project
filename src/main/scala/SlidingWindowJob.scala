import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.log4j.Logger
import com.typesafe.config.ConfigFactory

import scala.jdk.CollectionConverters.*

object SlidingWindowJob {
  private val logger = Logger.getLogger(getClass)
  private val config = ConfigFactory.load()
  private val windowSize = config.getInt("sliding-window.window-size")
  private val jobName = config.getString("sliding-window.job-name")
  private val inputSplitDelimiter = config.getString("sliding-window.input-split-delimiter")
  private val tokenIndex = config.getInt("sliding-window.token-index")

  private class SlidingWindowMapper extends Mapper[LongWritable, Text, Text, Text] {
    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      val inputLine = value.toString
      logger.info(s"Processing input line: $inputLine")

      val parts = inputLine.split(inputSplitDelimiter)
      if (parts.length > tokenIndex) {
        val token = parts(tokenIndex)
        logger.info(s"Extracted token: $token")
        context.write(new Text("data"), new Text(token))
      } else {
        logger.warn(s"Skipping input line due to insufficient parts: $inputLine")
      }
    }
  }

  private class SlidingWindowReducer extends Reducer[Text, Text, Text, Text] {
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
    val job = Job.getInstance(conf, jobName)
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