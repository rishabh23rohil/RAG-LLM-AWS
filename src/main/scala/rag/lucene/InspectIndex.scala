package rag.lucene

import org.apache.lucene.store.FSDirectory
import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.document.Document
import java.nio.file.Paths

object InspectIndex {
  def main(args: Array[String]): Unit = {
    if (args.length != 1) {
      System.err.println("usage: rag.lucene.InspectIndex <indexDir>")
      System.exit(1)
    }
    val indexDir = args(0)
    val dir     = FSDirectory.open(Paths.get(indexDir))
    val reader  = DirectoryReader.open(dir)
    try {
      println(s"Index: $indexDir")
      println(s"numDocs=${reader.numDocs()}, maxDoc=${reader.maxDoc()}, hasDeletions=${reader.hasDeletions}")
      if (reader.numDocs() > 0) {
        val doc: Document = reader.document(0)
        val fields = doc.getFields().toArray.map(_.asInstanceOf[org.apache.lucene.index.IndexableField])
        println("First doc stored fields:")
        fields
          .filter(_.fieldType().stored())
          .foreach { f =>
            val v = Option(doc.get(f.name())).getOrElse("<null>")
            println(s"  ${f.name()} = $v")
          }
      }
    } finally {
      reader.close()
    }
  }
}