ThisBuild / scalaVersion := "3.5.1"

lazy val root = (project in file("."))
  .settings(
    name := "rag-app",
    libraryDependencies ++= Seq(
      // Lucene core + analysis
      "org.apache.lucene" % "lucene-core" % "9.10.0",
      "org.apache.lucene" % "lucene-analysis-common" % "9.10.0",

      // PDF extraction
      "org.apache.pdfbox" % "pdfbox" % "2.0.31",

      // HTTP client + JSON (Ollama API)
      "com.softwaremill.sttp.client3" %% "core"  % "3.9.5",
      "com.softwaremill.sttp.client3" %% "circe" % "3.9.5",
      "io.circe" %% "circe-generic" % "0.14.9",
      "io.circe" %% "circe-parser"  % "0.14.9",

      // Logging
      "ch.qos.logback" % "logback-classic" % "1.5.6",

      // Testing
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    )
  )
