import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import java.net.URL
import java.io.{BufferedReader, InputStreamReader, BufferedWriter, OutputStreamWriter}

object TextShardingJob {
  def runJob(inputPath: String): Unit = {
    // Configurable size in bytes (e.g., 1MB = 1048576 bytes)
    val shardSize = 10000000 // Configure this value to your preferred size

    // HDFS setup
    val hadoopConf = new Configuration()
    val hdfs = FileSystem.get(hadoopConf)
    val hdfsPathPrefix = s"${inputPath}/wikitext_shard_"

    // Download file from URL
    val url = new URL("https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip")
    val connection = url.openConnection()
    val inputStream = connection.getInputStream

    // Read and process file
    val reader = new BufferedReader(new InputStreamReader(inputStream))
    var shardNum = 0
    var currentShardSize = 0L
    var writer: BufferedWriter = null

    reader.lines().forEach { line =>
      // If shard is null or exceeds size, create a new shard
      if (writer == null || currentShardSize > shardSize) {
        if (writer != null) writer.close()
        writer = new BufferedWriter(new OutputStreamWriter(hdfs.create(new Path(hdfsPathPrefix + shardNum + ".txt"))))
        shardNum += 1
        currentShardSize = 0L
      }

      // Write line and update size
      writer.write(line)
      writer.newLine()
      currentShardSize += line.getBytes.length + System.lineSeparator().getBytes.length
    }

    // Close resources
    if (writer != null) writer.close()
    reader.close()
    hdfs.close()

    println(s"Dataset sharded into $shardNum files based on size.")
  }
}
