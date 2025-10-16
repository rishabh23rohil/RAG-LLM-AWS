package rag

object ConfigCheck {
  def main(args: Array[String]): Unit = {
    val c = Cfg.load()
    val s =
      s"""CFG OK
         |inputList = ${c.inputList}
         |outIndex  = ${c.outIndex}
         |embedModel= ${c.embedModel}
         |chunk     = max=${c.chunkMax} overlap=${c.chunkOverlap}
         |sim       = ${c.sim}""".stripMargin
    println(s)
  }
}