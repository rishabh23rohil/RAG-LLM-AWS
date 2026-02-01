package rag.indexing

import io.circe.parser.decode
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{IntWritable, Text}
import org.apache.hadoop.mapreduce.Reducer
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.document._
import org.apache.lucene.index.{IndexWriter, IndexWriterConfig, VectorSimilarityFunction}
import org.apache.lucene.store.FSDirectory
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path => LocalPath}
import java.util.Comparator
import scala.jdk.CollectionConverters._
import scala.util.{Try, Success, Failure}

/**
 * Constructs HNSW vector index partitions from document segments.
 *
 * Each reducer instance builds one partition of the distributed index:
 * - Creates temporary local Lucene directory
 * - Indexes incoming document vectors with HNSW graph
 * - Persists index files to distributed storage
 * - Cleans up local resources
 */
class IndexPartitionBuilder extends Reducer[IntWritable, Text, Text, Text] {

  private val logger = LoggerFactory.getLogger(classOf[IndexPartitionBuilder])

  case class SegmentRecord(documentKey: String, segmentIndex: Int, content: String, vector: Vector[Float])

  private def parseRecord(jsonStr: String): Either[String, SegmentRecord] = {
    import io.circe.generic.auto._
    decode[SegmentRecord](jsonStr).left.map(_.getMessage)
  }

  private def transferToDistributedStorage(localDir: LocalPath, targetPath: Path, fs: FileSystem): (Int, Long) = {
    logger.debug(s"[Builder] Transferring index to ${targetPath}")
    
    Files.list(localDir).iterator().asScala.foldLeft((0, 0L)) { 
      case ((count, totalSize), filePath) =>
        val fileName = filePath.getFileName.toString
        val fileBytes = Files.size(filePath)
        
        val destPath = new Path(targetPath, fileName)
        val outputStream = fs.create(destPath, true)
        
        try {
          Files.copy(filePath, outputStream)
          (count + 1, totalSize + fileBytes)
        } finally {
          outputStream.close()
        }
    }
  }

  private def selectSimilarityFunction(name: String): VectorSimilarityFunction = name.toUpperCase match {
    case "EUCLIDEAN" => VectorSimilarityFunction.EUCLIDEAN
    case "DOT_PRODUCT" => VectorSimilarityFunction.DOT_PRODUCT
    case _ => VectorSimilarityFunction.COSINE
  }

  override def reduce(key: IntWritable, values: java.lang.Iterable[Text], context: Reducer[IntWritable, Text, Text, Text]#Context): Unit = {
    val partitionId = key.get
    val buildStart = System.currentTimeMillis()
    
    logger.info(s"[Builder] Starting partition $partitionId construction")

    // Initialize temporary index directory
    val tempIndexDir: LocalPath = Files.createTempDirectory(s"index-partition-$partitionId")
    logger.debug(s"[Builder] Temp directory: $tempIndexDir")

    // Configure Lucene writer
    val analyzer = new StandardAnalyzer()
    val writerConfig = new IndexWriterConfig(analyzer)
    val indexWriter = new IndexWriter(FSDirectory.open(tempIndexDir), writerConfig)

    val similarityFn = selectSimilarityFunction(
      context.getConfiguration.get("rag.similarity", "COSINE")
    )
    logger.debug(s"[Builder] Using similarity: $similarityFn")

    // Process incoming records
    var indexedCount = 0
    var errorCount = 0

    values.asScala.foreach { record =>
      parseRecord(record.toString) match {
        case Right(seg) =>
          val luceneDoc = new Document()
          luceneDoc.add(new StringField("document_key", seg.documentKey, Field.Store.YES))
          luceneDoc.add(new StringField("segment_index", seg.segmentIndex.toString, Field.Store.YES))
          luceneDoc.add(new TextField("content", seg.content, Field.Store.YES))
          luceneDoc.add(new KnnFloatVectorField("vector", seg.vector.toArray, similarityFn))

          indexWriter.addDocument(luceneDoc)
          indexedCount += 1

          if (indexedCount % 250 == 0) {
            logger.debug(s"[Builder] Progress: $indexedCount documents indexed")
          }

        case Left(errorMsg) =>
          logger.warn(s"[Builder] Parse error: $errorMsg")
          context.getCounter("index_builder", "parse_errors").increment(1)
          errorCount += 1
      }
    }

    // Finalize index
    Try(indexWriter.commit()) match {
      case Success(_) => logger.debug("[Builder] Index committed")
      case Failure(ex) => 
        logger.error(s"[Builder] Commit failed: ${ex.getMessage}", ex)
        throw ex
    }
    indexWriter.close()

    logger.info(s"[Builder] Partition $partitionId: indexed $indexedCount documents")

    // Transfer to distributed storage
    val outputRoot = FileOutputFormat.getOutputPath(context)
    val partitionPath = new Path(outputRoot, s"partition_$partitionId")
    
    val fs = partitionPath.getFileSystem(context.getConfiguration)
    fs.mkdirs(partitionPath)

    val (fileCount, totalBytes) = transferToDistributedStorage(tempIndexDir, partitionPath, fs)
    logger.debug(s"[Builder] Transferred $fileCount files ($totalBytes bytes)")

    // Cleanup temporary files
    Try {
      Files.walk(tempIndexDir)
        .sorted(Comparator.reverseOrder())
        .forEach(Files.delete)
    }.recover {
      case ex => logger.warn(s"[Builder] Cleanup warning: ${ex.getMessage}")
    }

    // Output summary
    val buildDuration = System.currentTimeMillis() - buildStart
    val summary = s"docs=$indexedCount,errors=$errorCount,time=${buildDuration}ms"
    context.write(new Text(s"partition_$partitionId"), new Text(summary))
    
    context.getCounter("index_builder", s"partition_${partitionId}_docs").increment(indexedCount)
    
    logger.info(s"[Builder] Partition $partitionId complete: $indexedCount docs in ${buildDuration}ms")
  }
}
