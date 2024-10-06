import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.mockito.MockitoSugar
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Mapper, Reducer}
import org.mockito.Mockito._
import org.mockito.ArgumentMatchers._
import com.typesafe.config.ConfigFactory
import org.yaml.snakeyaml.Yaml

import scala.jdk.CollectionConverters._

class StatisticsCollaterJobSpec extends AnyFlatSpec with Matchers with MockitoSugar {

  "StatisticsCollaterJob" should "initialize with correct configuration" in {
    noException should be thrownBy {
      StatisticsCollaterJob
    }
  }

  "TokenizationMapper" should "extract and emit token data correctly" in {
    val mapper = new StatisticsCollaterJob.TokenizationMapper()
    val context = mock[Mapper[LongWritable, Text, Text, Text]#Context]

    val inputLine = "word\t12345\t10"
    mapper.map(new LongWritable(0), new Text(inputLine), context)

    verify(context).write(new Text("12345"), new Text("word\t10"))
  }

  it should "handle malformed input" in {
    val mapper = new StatisticsCollaterJob.TokenizationMapper()
    val context = mock[Mapper[LongWritable, Text, Text, Text]#Context]

    val inputLine = "malformedInput"
    mapper.map(new LongWritable(0), new Text(inputLine), context)

    verify(context, never()).write(any[Text], any[Text])
  }

  "SimilarityMapper" should "extract and emit similarity data correctly" in {
    val mapper = new StatisticsCollaterJob.SimilarityMapper()
    val context = mock[Mapper[LongWritable, Text, Text, Text]#Context]

    val inputLine = "12345\tsimilar1(0.9),similar2(0.8)"
    mapper.map(new LongWritable(0), new Text(inputLine), context)

    verify(context).write(new Text("12345"), new Text("similar1(0.9),similar2(0.8)"))
  }

  it should "handle malformed input" in {
    val mapper = new StatisticsCollaterJob.SimilarityMapper()
    val context = mock[Mapper[LongWritable, Text, Text, Text]#Context]

    val inputLine = "malformedInput"
    mapper.map(new LongWritable(0), new Text(inputLine), context)

    verify(context, never()).write(any[Text], any[Text])
  }

  "StatisticsReducer" should "combine tokenization and similarity data correctly" in {
    val reducer = new StatisticsCollaterJob.StatisticsReducer()
    val context = mock[Reducer[Text, Text, Text, Text]#Context]

    val token = new Text("12345")
    val values = List(
      new Text("word\t10"),
      new Text("similar1(0.9),similar2(0.8)")
    ).asJava

    reducer.reduce(token, values, context)

    verify(context).write(isNull(), any[Text])
  }

  it should "handle missing tokenization or similarity data" in {
    val reducer = new StatisticsCollaterJob.StatisticsReducer()
    val context = mock[Reducer[Text, Text, Text, Text]#Context]

    val token = new Text("12345")
    val values = List(new Text("word\t10")).asJava

    reducer.reduce(token, values, context)

    verify(context).write(isNull(), any[Text])
  }
}