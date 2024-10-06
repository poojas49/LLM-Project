## Name : Pooja Chandrakant Shinde
## Email : pshin8@uic.edu

# Text Processing Pipeline

This project implements a MapReduce job for processing text data using AWS EMR and S3.

## Setup

1. Ensure you have an AWS account and access to EMR and S3 services.
2. Upload your input data to S3: `s3://llmpooja/input/`
3. Prepare your JAR file and upload it to S3: `s3://llmpooja/jars/your-application.jar`

## Running the Job

1. Create an EMR cluster using the AWS Management Console.
2. Add a Spark application step with the following configuration:
    - Application location: `s3://llmpooja/jars/your-application.jar`
    - Arguments:
      ```
      --class TextProcessingPipeline
      s3://llmpooja/input/
      s3://llmpooja/output/
      ```

## Output

The job output will be available in: `s3://llmpooja/output/tokenization/`

## Troubleshooting

If you encounter issues:
- Check EMR step logs for error messages
- Verify input data location and permissions
- Ensure your JAR includes all necessary dependencies

Remember to terminate your EMR cluster after job completion.