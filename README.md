## Name : Pooja Chandrakant Shinde
## Email : pshin8@uic.edu

## Videos
Job Running in Hadoop Pseudo Distributed Mode - https://youtu.be/yr6GhDT-DkY
Job Running in AWS EMR - https://youtu.be/3n63BBl1G-I

# Text Processing Pipeline

This project implements a pipeline which first shards the data based on the configured size in Hadoop HDFS (configured in application.conf file).
Then for each Split/Shard of Data it runs the Map reduce jobs which are:
1. TokenizationJob
2. SlidingWindowJob
3. EmbeddingJob
4. SemanticSimilarityJob
5. StatisticsCollaterJob
The role of each Map reduce job is explained and comments in the code.

## Setup

1. Install Java 8 - brew install homebrew/cask-versions/adoptopenjdk8
2. Install Scala 3.5.0 - brew install sbt - then install scala plugin from IntelliJ
3. Install Hadoop 3.4.0 - brew install hadoop
4. To configure Hadoop to run in pseudo distributed mode follow - https://github.com/0x1DOCD00D/CS441_Fall2024/blob/main/Homeworks/MapReduceHadoopExampleProgram.md

## Running the Job

1. Clone the Project.
2. run `sbt update` - to load all the dependencies
3. run `sbt clean compile` - to build the project
4. run `SBT_OPTS="-Xmx2G" sbt assembly` - to build a fat jar
5. run `hadoop namenode -Format` - to format and initialize the namenode
6. run `start-dfs.sh` - to start data and name nodes
7. run `start-yarn.sh` - to start resource managers
8. run `hdfs dfs -mkdir -p input` - to create input folder in hdfs
9. run `hdfs dfs -put /path/to/input/data/file /hdfs/input/folder/path
10. run `hadoop jar /path/to/fat/jar/file /hdfs/input/folder/path /hdfs/output/folder/path`

## Output

The job output will be available in: `/hdfs/output/folder/path`

## Troubleshooting

If you encounter issues:
- Check logs for error messages
- Verify input data location and permissions
- Ensure your JAR includes all necessary dependencies
