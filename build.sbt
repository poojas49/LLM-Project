import scala.collection.Seq

ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.5.0"

lazy val root = (project in file("."))
  .settings(
    name := "LLM-Project"
  )

libraryDependencies ++= Seq(
  "org.apache.hadoop" % "hadoop-common" % "3.3.6",
  "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.3.6",
  "ch.qos.logback" % "logback-classic" % "1.2.11", // For Logback
  "org.slf4j" % "slf4j-api" % "2.0.12", // For SLF4J
  "com.typesafe" % "config" % "1.4.3", // For Typesafe Config
  "com.knuddels" % "jtokkit" % "1.1.0",
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta7",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta7",
  "org.yaml" % "snakeyaml" % "2.0",
  // Testing libraries
  "org.scalatest" %% "scalatest" % "3.2.18" % Test,
  "org.scalatestplus" %% "mockito-3-4" % "3.2.10.0" % Test
)

assembly / assemblyMergeStrategy := {
  case PathList("javax", "xml", "bind", xs @ _*) => MergeStrategy.first
  case PathList("jakarta", "xml", "bind", xs @ _*) => MergeStrategy.first
  case PathList("META-INF", "services", xs @ _*)  => MergeStrategy.concat
  case PathList("META-INF", xs @ _*)              => MergeStrategy.discard
  case _                                          => MergeStrategy.first
}