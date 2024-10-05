import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.log4j.Logger
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.api.ndarray.INDArray
import com.typesafe.config.ConfigFactory

import scala.jdk.CollectionConverters._
import scala.util.{Try, Success, Failure}

object EmbeddingJob {
  private val logger = Logger.getLogger(getClass)
  private val config = ConfigFactory.load()

  private val windowSize = config.getInt("embedding-job.window-size")
  private val embeddingSize = config.getInt("embedding-job.embedding-size")
  private val jobName = config.getString("embedding-job.job-name")
  private val inputSplitDelimiter = config.getString("embedding-job.input-split-delimiter")

  import org.apache.hadoop.io.{LongWritable, Text}
  import org.apache.hadoop.mapreduce.{Mapper, Reducer}
  import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
  import org.nd4j.linalg.factory.Nd4j

  class EmbeddingMapper extends Mapper[LongWritable, Text, Text, Text] {
    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      val parts = value.toString.split(inputSplitDelimiter)
      parts match {
        case Array(keyType, tokens) =>
          val newKey = keyType.split("_")(0)
          context.write(new Text(newKey), new Text(tokens))
        case _ =>
          logger.warn(s"Malformed input: ${value.toString}")
      }
    }
  }

  class EmbeddingReducer extends Reducer[Text, Text, Text, Text] {
    private lazy val model: MultiLayerNetwork = createModel()

    override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      val allTokens = values.asScala.flatMap(_.toString.split(",").map(_.toInt)).toList

      val tokenEmbeddings = processTokens(allTokens)
      writeEmbeddings(tokenEmbeddings, context)
    }

    private def processTokens(tokens: List[Int]): Map[Int, List[Array[Float]]] = {
      tokens.sliding(windowSize).foldLeft(Map.empty[Int, List[Array[Float]]]) { (acc, window) =>
        if (window.length == windowSize) {
          val inputFeature = createInputFeature(window)
          model.fit(inputFeature, inputFeature)

          window.zipWithIndex.foldLeft(acc) { case (innerAcc, (token, index)) =>
            val embedding = model.getLayer(0).getParam("W").getColumn(index).toFloatVector
            innerAcc.updated(token, embedding :: innerAcc.getOrElse(token, List.empty))
          }
        } else acc
      }
    }

    private def writeEmbeddings(tokenEmbeddings: Map[Int, List[Array[Float]]], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      tokenEmbeddings.foreach { case (token, embeddings) =>
        val avgEmbedding = embeddings.transpose.map(v => v.sum / v.length)
        val embeddingStr = avgEmbedding.mkString("[", ", ", "]")
        context.write(new Text(token.toString), new Text(embeddingStr))
      }
    }

    private def createModel(): MultiLayerNetwork = {
      val conf = new NeuralNetConfiguration.Builder()
        .updater(new Adam())
        .list()
        .layer(new DenseLayer.Builder().nIn(windowSize).nOut(embeddingSize)
          .activation(Activation.fromString(config.getString("neural-network.activation-function")))
          .build())
        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.valueOf(config.getString("neural-network.loss-function")))
          .nIn(embeddingSize).nOut(windowSize)
          .activation(Activation.fromString(config.getString("neural-network.output-activation-function")))
          .build())
        .build()

      val network = new MultiLayerNetwork(conf)
      network.init()
      logger.info("Model created and initialized")
      network
    }

    private def createInputFeature(window: Seq[Int]): INDArray = {
      val floatArray = window.map(_.toFloat).toArray
      Nd4j.create(floatArray).reshape(1, windowSize)
    }
  }

  def runJob(conf: Configuration, input: Path, output: Path): Try[Unit] = {
    val setupJob = Try(Job.getInstance(conf, jobName)).map { job =>
      job.setJarByClass(this.getClass)
      job.setMapperClass(classOf[EmbeddingMapper])
      job.setReducerClass(classOf[EmbeddingReducer])
      job.setMapOutputKeyClass(classOf[Text])
      job.setMapOutputValueClass(classOf[Text])
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
        logger.info(s"Starting Embedding job with input: $input and output: $output")
        setupJob.get.waitForCompletion(true)
      }.flatMap { completed =>
        if (completed) Success(())
        else Failure(new Exception("Embedding job failed"))
      }
    }

    runAndComplete.map { _ =>
      logger.info("Embedding job completed successfully")
    }
  }
}