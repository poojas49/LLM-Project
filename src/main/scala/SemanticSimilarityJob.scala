import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.log4j.Logger
import com.typesafe.config.ConfigFactory

import scala.jdk.CollectionConverters._
import scala.util.{Try, Success, Failure}

object SemanticSimilarityJob {
  private val logger = Logger.getLogger(getClass)
  private val config = ConfigFactory.load()

  private val jobName = config.getString("semantic-similarity.job-name")
  private val inputSplitDelimiter = config.getString("semantic-similarity.input-split-delimiter")
  private val embeddingDelimiter = config.getString("semantic-similarity.embedding-delimiter")
  private val topK = config.getInt("semantic-similarity.top-k")
  private val similarityFormat = config.getString("semantic-similarity.similarity-format")

  class SimilarityMapper extends Mapper[LongWritable, Text, Text, Text] {
    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      value.toString.split(inputSplitDelimiter) match {
        case Array(token, embedding) =>
          context.write(new Text("all"), new Text(s"$token$embeddingDelimiter$embedding"))
        case _ =>
          logger.warn(s"Malformed input: ${value.toString}")
      }
    }
  }

  class SimilarityReducer extends Reducer[Text, Text, Text, Text] {
    override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      val embeddings = parseEmbeddings(values)
      val similarities = calculateSimilarities(embeddings)
      writeSimilarities(similarities, context)
    }

    private def parseEmbeddings(values: java.lang.Iterable[Text]): Map[String, Array[Float]] = {
      values.asScala.foldLeft(Map.empty[String, Array[Float]]) { (acc, value) =>
        value.toString.split(s"\\$embeddingDelimiter") match {
          case Array(token, embeddingStr) =>
            val embedding = embeddingStr.stripPrefix("[").stripSuffix("]").split(",").map(_.trim.toFloat)
            acc + (token -> embedding)
          case _ => acc
        }
      }
    }

    private def calculateSimilarities(embeddings: Map[String, Array[Float]]): Map[String, Seq[(String, Double)]] = {
      embeddings.map { case (token1, embedding1) =>
        val tokenSimilarities = embeddings.filter(_._1 != token1).map { case (token2, embedding2) =>
          (token2, cosineSimilarity(embedding1, embedding2))
        }.toSeq.sortBy(-_._2).take(topK)
        (token1, tokenSimilarities)
      }
    }

    private def writeSimilarities(similarities: Map[String, Seq[(String, Double)]], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      similarities.foreach { case (token, similarTokens) =>
        val formattedSimilarities = similarTokens.map { case (similarToken, similarity) =>
          s"$similarToken(${similarity.formatted(similarityFormat)})"
        }.mkString(",")
        context.write(new Text(token), new Text(formattedSimilarities))
      }
    }

    private def cosineSimilarity(vec1: Array[Float], vec2: Array[Float]): Double = {
      require(vec1.length == vec2.length, "Vectors must have the same length")
      val (dotProduct, mag1, mag2) = (vec1 zip vec2).foldLeft((0.0, 0.0, 0.0)) {
        case ((dot, mag1Sq, mag2Sq), (v1, v2)) =>
          (dot + v1 * v2, mag1Sq + v1 * v1, mag2Sq + v2 * v2)
      }
      dotProduct / (math.sqrt(mag1) * math.sqrt(mag2))
    }
  }

  def runJob(conf: Configuration, input: Path, output: Path): Try[Unit] = {
    val setupJob = Try(Job.getInstance(conf, jobName)).map { job =>
      job.setJarByClass(this.getClass)
      job.setMapperClass(classOf[SimilarityMapper])
      job.setReducerClass(classOf[SimilarityReducer])
      job.setOutputKeyClass(classOf[Text])
      job.setOutputValueClass(classOf[Text])
      FileInputFormat.addInputPath(job, input)
      FileOutputFormat.setOutputPath(job, output)
      job
    }

    val cleanupOutput = setupJob.flatMap(_ => Try(FileSystem.get(conf))).map { fs =>
      if (fs.exists(output)) fs.delete(output, true)
    }

    val runAndComplete = cleanupOutput.flatMap { _ =>
      Try {
        logger.info(s"Starting $jobName job with input: $input and output: $output")
        setupJob.get.waitForCompletion(true)
      }.flatMap { completed =>
        if (completed) Success(())
        else Failure(new Exception(s"$jobName job failed"))
      }
    }

    runAndComplete.map { _ =>
      logger.info(s"$jobName job completed successfully")
    }
  }
}