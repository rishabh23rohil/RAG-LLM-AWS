package rag
import com.typesafe.config.ConfigFactory

final case class AppCfg(
  inputList: String,
  outIndex: String,
  embedModel: String,
  chunkMax: Int,
  chunkOverlap: Int,
  sim: String
)

object Cfg {
  private val root = ConfigFactory.load().getConfig("app")
  def load(): AppCfg =
    AppCfg(
      inputList    = root.getString("inputList"),
      outIndex     = root.getString("outIndex"),
      embedModel   = root.getString("embedModel"),
      chunkMax     = root.getConfig("chunk").getInt("maxChars"),
      chunkOverlap = root.getConfig("chunk").getInt("overlap"),
      sim          = root.getConfig("knn").getString("similarity")
    )
}