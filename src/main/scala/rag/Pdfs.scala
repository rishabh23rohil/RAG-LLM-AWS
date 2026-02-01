package rag

import java.io.InputStream
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.PDFTextStripper
import java.nio.file.Path

object Pdfs {

  // Original version (still useful for local files)
  def readText(path: Path): String = {
    val doc = PDDocument.load(path.toFile)
    try {
      val stripper = new PDFTextStripper()
      stripper.getText(doc)
    } finally {
      doc.close()
    }
  }

  // New version for InputStream (S3 / HDFS / remote)
  def readText(inputStream: InputStream): String = {
    val doc = PDDocument.load(inputStream)
    try {
      val stripper = new PDFTextStripper()
      stripper.getText(doc)
    } finally {
      doc.close()
    }
  }
}
