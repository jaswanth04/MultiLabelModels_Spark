package evaluation

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{udf,col}
import org.apache.spark.sql.DataFrame

trait MultiLabelModelEvaluation {

  def getEvaluationMetrics(df: DataFrame, responseVariables: Seq[String]): MultiLabelMetrics = {

    val outColName = "flagVector"
    val vectorizedDf = Seq("", "_prediction")
      .map(a => (responseVariables.map(_.concat(a)), outColName.concat(a)))
      .map(x => new VectorAssembler().setInputCols(x._1.toArray).setOutputCol(x._2))
      .foldLeft(df)((df,vec) => vec.transform(df))
      .select(outColName, outColName.concat("_prediction"))
    vectorizedDf.cache()

    val n = vectorizedDf.count().toDouble
    val l = responseVariables.length

    val mod_yAndyhat = udf((a: Vector, b: Vector) => a.toArray.map(_.toInt).zip(b.toArray.map(_.toInt)).map(x=> x._1 & x._2).sum)
    val mod_yOryhat = udf((a: Vector, b: Vector) => a.toArray.map(_.toInt).zip(b.toArray.map(_.toInt)).map(x=> x._1 | x._2).sum)
    val mod = udf((a:Vector) => a.toArray.sum)
    val correctPredictionCount = udf((a: Vector, b: Vector) => a.toArray.zip(b.toArray).map(x => x._1 == x._2).count(x => x))
    val div = udf((a:Int, b: Double) => if (a == 0) 0 else a/b)

    val modifiedDf = vectorizedDf.withColumn("mod_yAndyhat", mod_yAndyhat(col("flagVector"), col("flagVector_prediction")))
      .withColumn("mod_y", mod(col("flagVector")))
      .withColumn("mod_yHat", mod(col("flagVector_prediction")))
      .withColumn("mod_yOryhat", mod_yOryhat(col("flagVector"), col("flagVector_prediction")))
      .withColumn("correctPredictionCount", correctPredictionCount(col("flagVector"), col("flagVector_prediction")))

    val calcDf = modifiedDf.withColumn("yAndyHat_by_yOryHat", div(modifiedDf("mod_yAndyhat"), modifiedDf("mod_yOryhat")))
      .withColumn("yAndyHat_by_yHat", div(modifiedDf("mod_yAndyhat"),modifiedDf("mod_yHat")))
      .withColumn("yAndyHat_by_y", div(modifiedDf("mod_yAndyhat"), modifiedDf("mod_y")))
      .withColumn("yAndyHat_by_y_plus_yHat", div(modifiedDf("mod_yAndyhat"), modifiedDf("mod_yHat") + modifiedDf("mod_y")))

    val requiredColumnsToSum = List("yAndyHat_by_yOryHat", "yAndyHat_by_yHat", "yAndyHat_by_y", "yAndyHat_by_y_plus_yHat", "correctPredictionCount")

    val x = calcDf.groupBy().sum(requiredColumnsToSum: _*).first()

    val hammingLoss = 1 - (x.getLong(4)/(n*l))
    val subsetLoss = 1 - (vectorizedDf.filter(col("flagVector") === col("flagVector_prediction")).count()/n)
    val accuracy = x.getDouble(0)/n
    val precision = x.getDouble(1)/n
    val recall = x.getDouble(2)/n
    val f1 = 2*x.getDouble(3)/n

    MultiLabelMetrics(hammingLoss, subsetLoss, accuracy, precision, recall, f1)
  }

}
