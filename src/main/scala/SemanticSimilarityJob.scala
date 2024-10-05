import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.log4j.Logger
import com.typesafe.config.ConfigFactory

import scala.jdk.CollectionConverters._
import scala.collection.mutable

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
      val parts = value.toString.split(inputSplitDelimiter)
      if (parts.length == 2) {
        val token = parts(0)
        val embedding = parts(1)
        context.write(new Text("all"), new Text(s"$token$embeddingDelimiter$embedding"))
      }
    }
  }

  class SimilarityReducer extends Reducer[Text, Text, Text, Text] {
    override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      val embeddings = mutable.Map[String, Array[Float]]()

      // Parse embeddings
      values.asScala.foreach { value =>
        val parts = value.toString.split(s"\\$embeddingDelimiter")
        if (parts.length == 2) {
          val token = parts(0)
          val embedding = parts(1).stripPrefix("[").stripSuffix("]").split(",").map(_.trim.toFloat)
          embeddings(token) = embedding
        }
      }

      // Calculate cosine similarity for each pair of tokens
      embeddings.foreach { case (token1, embedding1) =>
        val similarities = embeddings.filter(_._1 != token1).map { case (token2, embedding2) =>
          (token2, cosineSimilarity(embedding1, embedding2))
        }.toSeq.sortBy(-_._2).take(topK)

        val similarTokens = similarities.map { case (token, similarity) =>
          s"$token(${similarity.formatted(similarityFormat)})"
        }.mkString(",")

        context.write(new Text(token1), new Text(similarTokens))
      }
    }

    private def cosineSimilarity(vec1: Array[Float], vec2: Array[Float]): Double = {
      require(vec1.length == vec2.length, "Vectors must have the same length")
      val dotProduct = (vec1 zip vec2).map { case (v1, v2) => v1 * v2 }.sum
      val magnitude1 = math.sqrt(vec1.map(x => x * x).sum)
      val magnitude2 = math.sqrt(vec2.map(x => x * x).sum)
      dotProduct / (magnitude1 * magnitude2)
    }
  }

  def runJob(conf: Configuration, input: Path, output: Path): Unit = {
    val job = Job.getInstance(conf, jobName)
    job.setJarByClass(this.getClass)
    job.setMapperClass(classOf[SimilarityMapper])
    job.setReducerClass(classOf[SimilarityReducer])
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])
    FileInputFormat.addInputPath(job, input)
    FileOutputFormat.setOutputPath(job, output)

    // Delete output path if it exists
    val fs = FileSystem.get(conf)
    if (fs.exists(output)) {
      fs.delete(output, true)
    }

    logger.info(s"Starting $jobName job with input: $input and output: $output")
    if (!job.waitForCompletion(true)) {
      logger.error(s"$jobName job failed")
      System.exit(1)
    }
    logger.info(s"$jobName job completed successfully")
  }
}