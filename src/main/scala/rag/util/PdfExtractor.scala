package rag.util

import java.io.InputStream
import java.nio.file.Path
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.PDFTextStripper
import org.slf4j.LoggerFactory
import scala.util.{Try, Using}

/**
 * PDF text extraction utilities.
 *
 * Supports extraction from:
 * - Local filesystem paths
 * - Input streams (for HDFS/S3 sources)
 */
object PdfExtractor {
  
  private val logger = LoggerFactory.getLogger(classOf[PdfExtractor.type])

  /**
   * Extracts text content from a local PDF file.
   */
  def extractText(filePath: Path): String = {
    logger.debug(s"[PdfExtractor] Loading local file: $filePath")
    
    Using.resource(PDDocument.load(filePath.toFile)) { document =>
      val textStripper = new PDFTextStripper()
      val content = textStripper.getText(document)
      logger.trace(s"[PdfExtractor] Extracted ${content.length} characters")
      content
    }
  }

  /**
   * Extracts text content from a PDF input stream.
   * Suitable for distributed storage sources (HDFS, S3).
   */
  def extractText(inputStream: InputStream): String = {
    logger.debug("[PdfExtractor] Loading from input stream")
    
    Using.resource(PDDocument.load(inputStream)) { document =>
      val textStripper = new PDFTextStripper()
      val content = textStripper.getText(document)
      logger.trace(s"[PdfExtractor] Extracted ${content.length} characters")
      content
    }
  }

  /**
   * Safely attempts text extraction, returning None on failure.
   */
  def tryExtract(filePath: Path): Option[String] = {
    Try(extractText(filePath)).toOption
  }
}
