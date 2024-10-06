import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.mockito.MockitoSugar
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Mapper, Reducer}
import org.mockito.Mockito._
import org.mockito.ArgumentMatchers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

import scala.jdk.CollectionConverters._

class EmbeddingJobSpec extends AnyFlatSpec with Matchers with MockitoSugar {

  "EmbeddingJob" should "initialize with correct configuration" in {
    noException should be thrownBy {
      EmbeddingJob
    }
  }

  "EmbeddingMapper" should "extract and emit tokens correctly" in {
    val mapper = new EmbeddingJob.EmbeddingMapper()
    val context = mock[Mapper[LongWritable, Text, Text, Text]#Context]

    val inputLine = "input_0\ttoken1,token2,token3"
    mapper.map(new LongWritable(0), new Text(inputLine), context)

    verify(context).write(new Text("input"), new Text("token1,token2,token3"))
  }

  it should "handle malformed input" in {
    val mapper = new EmbeddingJob.EmbeddingMapper()
    val context = mock[Mapper[LongWritable, Text, Text, Text]#Context]

    val inputLine = "malformedInput"
    mapper.map(new LongWritable(0), new Text(inputLine), context)

    verify(context, never()).write(any[Text], any[Text])
  }

  "EmbeddingReducer" should "process tokens and generate embeddings" in {
    val reducer = new EmbeddingJob.EmbeddingReducer()
    val context = mock[Reducer[Text, Text, Text, Text]#Context]

    val tokens = List("1,2,3", "2,3,4", "3,4,5").map(new Text(_)).asJava
    reducer.reduce(new Text("input"), tokens, context)

    // Verify that embeddings are written for each unique token
    verify(context, atLeastOnce()).write(any[Text], any[Text])
  }

  it should "handle empty input" in {
    val reducer = new EmbeddingJob.EmbeddingReducer()
    val context = mock[Reducer[Text, Text, Text, Text]#Context]

    val tokens = List.empty[Text].asJava
    reducer.reduce(new Text("input"), tokens, context)

    verify(context, never()).write(any[Text], any[Text])
  }

  "createModel" should "return a valid MultiLayerNetwork" in {
    val reducerInstance = new EmbeddingJob.EmbeddingReducer()
    val model = reducerInstance.createModel()

    model shouldBe a[MultiLayerNetwork]
    model.getLayers.length should be > 0
  }

  "runJob" should "execute without throwing exceptions" in {
    val conf = new org.apache.hadoop.conf.Configuration()
    val input = new org.apache.hadoop.fs.Path("test-input")
    val output = new org.apache.hadoop.fs.Path("test-output")

    noException should be thrownBy EmbeddingJob.runJob(conf, input, output)
  }
}