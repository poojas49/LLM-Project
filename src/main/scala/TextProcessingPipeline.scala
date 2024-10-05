import com.typesafe.config.ConfigFactory
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat

import scala.util.{Failure, Success, Try}
import scala.jdk.CollectionConverters._

object TextProcessingPipeline {
  def main(args: Array[String]): Unit = {
    args.toList match {
      case inputPath :: outputBasePath :: Nil =>
        processFiles(new Path(inputPath), new Path(outputBasePath))
      case _ =>
        System.err.println("Usage: TextProcessingPipeline <input path> <output path>")
        System.exit(-1)
    }
  }

  def processFiles(inputPath: Path, outputBasePath: Path): Unit = {
    val result = Try(ConfigFactory.load())
      .flatMap { config =>
        for {
          conf <- configureHadoop(config)
          output <- runPipeline(config, conf, inputPath, outputBasePath)
        } yield output
      }

    result match {
      case Success(output) => println(s"Pipeline completed. Final output: $output")
      case Failure(exception) =>
        System.err.println(s"Pipeline failed: ${exception.getMessage}")
        exception.printStackTrace()
    }
  }

  def configureHadoop(config: com.typesafe.config.Config): Try[Configuration] = Try {
    val conf = new Configuration()
    val dataShardSizeInMB = config.getLong("hadoop.split.size.mb")
    val numReducers = config.getInt("hadoop.num.reducers")
    conf.setLong(FileInputFormat.SPLIT_MAXSIZE, dataShardSizeInMB * 1024 * 1024)
    conf.setInt("mapreduce.job.reduces", numReducers)
    conf
  }

  def runPipeline(config: com.typesafe.config.Config, conf: Configuration, inputPath: Path, outputBasePath: Path): Try[Path] = {
    val jobsConfig = config.getConfigList("pipeline.jobs").asScala.toList
    val jobOutputs = jobsConfig.map { jobConfig =>
      val jobName = jobConfig.getString("name")
      val jobClass = jobConfig.getString("class")
      (jobName, Class.forName(jobClass).getMethod("runJob", classOf[Configuration], classOf[Path], classOf[Path]))
    }

    val pipelineResult = jobOutputs.foldLeft(Try(inputPath)) {
      case (prevOutputTry, (jobName, runJobMethod)) =>
        prevOutputTry.flatMap { prevOutput =>
          val output = new Path(outputBasePath, jobName)
          Try(runJobMethod.invoke(null, conf, prevOutput, output)).map(_ => output)
        }
    }

    pipelineResult.flatMap { _ =>
      val tokenizationOutput = new Path(outputBasePath, jobsConfig.head.getString("name"))
      val semanticSimilarityOutput = new Path(outputBasePath, jobsConfig.last.getString("name"))
      val finalJobConfig = config.getConfig("pipeline.final-job")
      val finalOutput = new Path(outputBasePath, finalJobConfig.getString("name"))
      conf.setInt("mapreduce.job.reduces", finalJobConfig.getInt("num.reducers"))
      Try(StatisticsCollaterJob.runJob(conf, tokenizationOutput, semanticSimilarityOutput, finalOutput))
        .map(_ => finalOutput)
    }
  }
}