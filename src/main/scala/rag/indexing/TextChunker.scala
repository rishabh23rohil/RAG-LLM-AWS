package rag.indexing

import org.slf4j.LoggerFactory

/**
 * Utility for segmenting documents into overlapping text windows.
 *
 * Designed for vector embedding pipelines where:
 * - Input length must respect model token limits
 * - Overlapping windows maintain semantic continuity
 * - Sentence-aware boundaries improve retrieval quality
 */
object TextChunker {
  
  private val logger = LoggerFactory.getLogger(classOf[TextChunker.type])

  /**
   * Cleans and normalizes input text.
   */
  def cleanText(input: String): String = {
    val cleaned = input.replaceAll("\\s+", " ").trim
    logger.trace(s"[Chunker] Cleaned: ${input.length} -> ${cleaned.length} chars")
    cleaned
  }

  /**
   * Segments text into overlapping windows with intelligent boundary detection.
   *
   * @param text Source text to segment
   * @param maxLength Maximum characters per segment (default: 1024)
   * @param overlap Characters shared between adjacent segments (default: 128)
   * @return Sequence of text segments
   */
  def segment(text: String, maxLength: Int = 1024, overlap: Int = 128): Vector[String] = {
    require(overlap < maxLength, s"Overlap ($overlap) must be smaller than maxLength ($maxLength)")
    
    val normalizedText = cleanText(text)
    
    if (normalizedText.isEmpty) return Vector.empty

    @scala.annotation.tailrec
    def buildSegments(startPos: Int, accumulated: Vector[String]): Vector[String] = {
      if (startPos >= normalizedText.length) {
        accumulated
      } else {
        val endPos = math.min(startPos + maxLength, normalizedText.length)
        val window = normalizedText.substring(startPos, endPos)

        // Prefer breaking at sentence boundaries
        val breakPoint = window.lastIndexWhere(c => c == '.' || c == '!' || c == '?' || c == '\n')
        
        val segment = if (breakPoint > (maxLength * 0.5).toInt) {
          window.substring(0, breakPoint + 1)
        } else {
          window
        }

        val nextStart = startPos + math.max(1, segment.length - overlap)
        buildSegments(nextStart, accumulated :+ segment.trim)
      }
    }

    val segments = buildSegments(0, Vector.empty)
    logger.debug(s"[Chunker] Produced ${segments.size} segments from ${normalizedText.length} chars")
    segments
  }

  /**
   * Alternative segmentation using paragraph boundaries.
   */
  def segmentByParagraph(text: String, maxLength: Int = 2048): Vector[String] = {
    val paragraphs = text.split("\n\n+").map(_.trim).filter(_.nonEmpty)
    
    paragraphs.foldLeft(Vector.empty[String]) { (acc, para) =>
      if (para.length <= maxLength) {
        acc :+ para
      } else {
        acc ++ segment(para, maxLength, overlap = 64)
      }
    }
  }
}
