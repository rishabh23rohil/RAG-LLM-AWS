ThisBuild / scalaVersion := "3.5.1"
ThisBuild / version      := "0.1.0"
ThisBuild / organization := "dev.local"

lazy val `rag-app` = (project in file("."))
  .settings(
    name := "rag-app",
    libraryDependencies ++= Seq(
      // PDF parsing
      "org.apache.pdfbox" % "pdfbox" % "2.0.30",

      // HTTP client (sttp)
      "com.softwaremill.sttp.client3" %% "core"           % "3.9.7",
      "com.softwaremill.sttp.client3" %% "okhttp-backend" % "3.9.7",
      // JSON codec for sttp (adds sttp.client3.circe.*)
      "com.softwaremill.sttp.client3" %% "circe"          % "3.9.7",

      // Circe core JSON library
      "io.circe" %% "circe-core"    % "0.14.7",
      "io.circe" %% "circe-generic" % "0.14.7",
      "io.circe" %% "circe-parser"  % "0.14.7",

      // Testing
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    ),
    // help some IDEs detect the test framework
    testFrameworks += new TestFramework("org.scalatest.tools.Framework")
  )
