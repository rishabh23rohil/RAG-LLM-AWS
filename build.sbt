// ===========================================================
// build.sbt for RAG-LLM-AWS Project
// Author: Rishabh Rohil
// ===========================================================

ThisBuild / scalaVersion := "3.5.1"

// -------------------- Dependencies -------------------------
libraryDependencies ++= Seq(
  // --- Lucene (vector search + analysis) ---
  "org.apache.lucene" % "lucene-core" % "9.10.0",
  "org.apache.lucene" % "lucene-analysis-common" % "9.10.0",

  // --- PDF extraction ---
  "org.apache.pdfbox" % "pdfbox" % "2.0.31",

  // --- HTTP + JSON client (for Ollama API) ---
  "com.softwaremill.sttp.client3" %% "core"  % "3.9.5",
  "com.softwaremill.sttp.client3" %% "circe" % "3.9.5",
  "io.circe" %% "circe-generic" % "0.14.9",
  "io.circe" %% "circe-parser"  % "0.14.9",

  // --- Config + Logging ---
  "com.typesafe" % "config" % "1.4.3",
  "ch.qos.logback" % "logback-classic" % "1.5.6",

  // --- Testing ---
  "org.scalatest" %% "scalatest" % "3.2.19" % Test
)

// -------------------- Hadoop (MapReduce API) ----------------
libraryDependencies ++= Seq(
  "org.apache.hadoop" % "hadoop-common" % "3.3.6" % "provided",
  "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.3.6" % "provided"
)

// -------------------- Assembly Plugin (fat jar) --------------
import sbtassembly.AssemblyPlugin.autoImport._

lazy val root = (project in file("."))
  .settings(
    name := "rag-app",
    version := "1.0",
    assembly / test := {},

    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case x if x.endsWith("module-info.class") => MergeStrategy.discard
      case _ => MergeStrategy.first
    },

    Compile / mainClass := Some("rag.lucene.KnnSearch")
  )

// ============================================================
// End of build.sbt
// ============================================================
