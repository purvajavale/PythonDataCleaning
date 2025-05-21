def main():
    bucket = "your-bucket-name"
    s3_input_key = "input-data/raw_data.csv"      # S3 key for input file
    local_input_path = "/tmp/raw_data.csv"        # local temp path to download input
    local_output_path = "/tmp/cleaned_data.csv"   # local temp path to save cleaned output
    s3_output_key = "processed-data/cleaned_data.csv"  # S3 key for cleaned output

    try:
        # Create S3 uploader instance
        uploader = S3Uploader(bucket)

        # Download file from S3
        print(uploader.download_file(s3_input_key, local_input_path))

        # Load data
        loader = CSVDataLoader(local_input_path)
        df = loader.load()

        # Clean data
        cleaner = DataCleaner(df)
        cleaner.handle_missing().remove_duplicates().remove_outliers()
        cleaner.correct_data_types().standardize_strings().scale_numeric()
        cleaned_df = cleaner.get_cleaned_data()

        # Save cleaned data locally
        saver = CSVDataSaver()
        saver.save(cleaned_df, local_output_path)

        # Upload cleaned data back to S3
        print(uploader.upload_file(local_output_path, s3_output_key))

        print("Data cleaning and upload completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

