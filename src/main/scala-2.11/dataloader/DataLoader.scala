package dataloader

import org.apache.spark.sql.DataFrame

abstract class DataLoader {

  def dataDf: DataFrame
  def getFinalDf: DataFrame
}
