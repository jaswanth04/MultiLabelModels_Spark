package dataloader

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession}

class FullTableData(fileName: String,
                    spark: SparkSession,
                    xIgnoreCols: Seq[String] = Seq(),
                    yIgnoreCols: Seq[String] = Seq()) extends DataLoader {

  val dataDf: DataFrame = spark.read.option("header","true").option("inferSchema","true").csv(fileName)

  val colNames: Array[String] = dataDf.schema.fieldNames

  def set_x_IgnoreCols(xIgnore: Seq[String]) = new FullTableData(this.fileName, this.spark, xIgnore, this.yIgnoreCols)
  def set_y_IgnoreCols(yIgnore: Seq[String]) = new FullTableData(this.fileName, this.spark, this.xIgnoreCols, yIgnore)

  val ignoreCols: Seq[String] = xIgnoreCols ++ yIgnoreCols
  val reqCols: Array[String] = colNames.filterNot(colName => ignoreCols contains colName)

  override def getFinalDf: DataFrame = dataDf.select(reqCols.map(col): _*)

}
