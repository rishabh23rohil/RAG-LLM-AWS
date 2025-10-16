package rag.stats

object Tokenize:
  private val wordRe = raw"[A-Za-z][A-Za-z0-9_\-']{1,}".r
  def words(s: String): Vector[String] =
    wordRe.findAllIn(s.toLowerCase).toVector
