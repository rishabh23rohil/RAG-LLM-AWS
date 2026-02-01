package rag

import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.Mapper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.slf4j.LoggerFactory
import java.nio.file.Paths

/**
 * MapReduce mapper for vocabulary analysis and embedding generation.
 *
 * Supports both distributed (HDFS/S3) and local filesystem execution.
 *
 * Processing steps:
 * 1. Load PDF from specified path (auto-detects storage type)
 * 2. Extract text content using PDFBox
 * 3. Clean and tokenize the extracted text
 * 4. Remove common stopwords
 * 5. Output (word, 1) tuples for aggregation
 */
class WordStatsMapper extends Mapper[LongWritable, Text, Text, IntWritable]:

  private val log = LoggerFactory.getLogger(getClass)

  private val stopwords = Set(
    "the", "is", "and", "a", "an", "in", "on", "of", "for", "to",
    "this", "that", "it", "with", "as", "by", "at", "from", "or", "be",
    "are", "was", "were", "has", "had", "have", "not", "but", "can", "could"
  )

  override def map(
                    key: LongWritable,
                    value: Text,
                    ctx: Mapper[LongWritable, Text, Text, IntWritable]#Context
                  ): Unit =
    val pathStr = value.toString.trim
    if (pathStr.isEmpty) then
      log.warn("Empty path string received")
      ctx.getCounter("rag", "empty_paths").increment(1)
      return

    log.info("-" * 80)
    log.info(s"Vocabulary analysis started for: $pathStr")

    val conf = ctx.getConfiguration
    val hadoopPath = new Path(pathStr)
    val docId = hadoopPath.getName

    // Step 1: Read the PDF content (determine source automatically)
    val maybeText: Either[String, String] =
      try
        if pathStr.startsWith("s3://") || pathStr.startsWith("s3a://") || pathStr.startsWith("hdfs://") then
          val fs = FileSystem.get(hadoopPath.toUri, conf)
          if !fs.exists(hadoopPath) then
            Left(s"Path not found: $pathStr")
          else
            log.info(s"Reading PDF from distributed FS: ${fs.getScheme}://${hadoopPath}")
            val inputStream = fs.open(hadoopPath)
            try Right(Pdfs.readText(inputStream))
            finally inputStream.close()
        else
          // Local file system mode
          log.info(s"Reading PDF from local path: $pathStr")
          Right(Pdfs.readText(Paths.get(pathStr)))
      catch
        case e: Exception =>
          log.error(s"Failed to read PDF $docId: ${e.getMessage}", e)
          Left(e.getMessage)

    // Handle missing or failed reads
    val text = maybeText match
      case Right(content) if content.trim.nonEmpty => content
      case Right(_) =>
        log.warn(s"No text extracted from $docId (empty or image-only PDF)")
        ctx.getCounter("rag", "empty_pdfs").increment(1)
        return
      case Left(err) =>
        log.error(s"Failed to read text for $docId: $err")
        ctx.getCounter("rag", "pdf_read_failures").increment(1)
        return

    // Step 2: Normalize text
    val cleanText = text.toLowerCase.replaceAll("[^a-z ]", " ")

    // Step 3: Tokenize and filter stopwords
    val words = cleanText
      .split("\\s+")
      .filter(w => w.nonEmpty && w.length >= 2 && !stopwords.contains(w))

    log.info(s"Extracted ${words.length} candidate words from $docId")

    // Step 4: Emit <word, 1> for each token
    words.foreach { word =>
      ctx.write(new Text(word), new IntWritable(1))
    }

    ctx.getCounter("wordstats", "processed_docs").increment(1)
    ctx.getCounter("wordstats", "emitted_words").increment(words.length)
    log.info(s"Completed processing for $docId — ${words.length} words emitted.")
    log.info("=" * 80)
