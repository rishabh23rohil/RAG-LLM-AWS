ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.5.1"

lazy val root = (project in file("."))
  .settings(
    name := "RAG-LLM-AWS",
    version := "1.0.0",
    organization := "com.rishabhRohil"
  )

libraryDependencies ++= Seq(
  // Retrieval index
  "org.apache.lucene" % "lucene-core" % "9.10.0",
  "org.apache.lucene" % "lucene-analysis-common" % "9.10.0",

  // PDF extraction
  "org.apache.pdfbox" % "pdfbox" % "2.0.31",

  // HTTP + JSON
  "com.softwaremill.sttp.client3" %% "core"  % "3.9.5",
  "com.softwaremill.sttp.client3" %% "circe" % "3.9.5",
  "io.circe" %% "circe-generic" % "0.14.9",
  "io.circe" %% "circe-parser"  % "0.14.9",

  // Logging
  "ch.qos.logback" % "logback-classic" % "1.5.6",

  // Hadoop (for MapReduce)
  "org.apache.hadoop" % "hadoop-common" % "3.3.6",
  "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.3.6",

  // HTTP4s for REST API
  "org.http4s" %% "http4s-ember-server" % "0.23.23",
  "org.http4s" %% "http4s-dsl" % "0.23.23",
  "org.http4s" %% "http4s-circe" % "0.23.23",

  // Cats Effect
  "org.typelevel" %% "cats-effect" % "3.5.2",

  // IP4s for networking types
  "com.comcast" %% "ip4s-core" % "3.3.0",

  // Testing frameworks
  "org.scalatest" %% "scalatest" % "3.2.17" % Test,
  "org.scalatestplus" %% "mockito-4-11" % "3.2.17.0" % Test,
  "org.mockito" % "mockito-core" % "5.7.0" % Test,

  // Hadoop testing
  "org.apache.hadoop" % "hadoop-minicluster" % "3.3.6" % Test,
  "org.apache.hadoop" % "hadoop-hdfs" % "3.3.6" % Test classifier "tests",
  "org.apache.hadoop" % "hadoop-common" % "3.3.6" % Test classifier "tests",

  // HTTP testing
  "org.http4s" %% "http4s-client" % "0.23.23" % Test
)

import sbtassembly.AssemblyPlugin.autoImport._

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", "services", xs @ _*) => MergeStrategy.concat
  case PathList("META-INF", xs @ _*)             => MergeStrategy.discard
  case "module-info.class"                       => MergeStrategy.discard
  case PathList("javax", "xml", "bind", xs @ _*) => MergeStrategy.first
  case PathList("jakarta", "xml", "bind", xs @ _*) => MergeStrategy.first
  case PathList("org", "slf4j", xs @ _*)         => MergeStrategy.first
  case PathList("com", "fasterxml", "jackson", xs @ _*) => MergeStrategy.first
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}
