package evaluation

import org.apache.spark.sql.{DataFrame, Row}

trait IndividualModelEvaluation {

  def evaluate(df: DataFrame, actualColumnName: String, predictedColumnName: String): ConfusionMatrix = {
    val predictionMetricValues: Array[Row] = df.groupBy(actualColumnName, predictedColumnName).count().collect()

    val truePositives = predictionMetricValues.filter(r => (r(0) == 1) & (r(1) == 1)).head(2).toString.toInt
    val trueNegatives = predictionMetricValues.filter(r => (r(0) == 0) & (r(1) == 0)).head(2).toString.toInt
    val falsePositives = predictionMetricValues.filter(r => (r(0) == 0) & (r(1) == 1)).head(2).toString.toInt
    val falseNegatives = predictionMetricValues.filter(r => (r(0) == 1) & (r(1) == 0)).head(2).toString.toInt

    ConfusionMatrix(truePositives, trueNegatives, falsePositives, falseNegatives)

  }

  def accuracy(cf: ConfusionMatrix): Double = (cf.truePositives.toDouble +
    cf.trueNegatives)/(cf.truePositives +
    cf.trueNegatives +
    cf.falseNegatives +
    cf.falsePositives)

  def precision(cf: ConfusionMatrix): Double = cf.truePositives.toDouble/(cf.truePositives + cf.falsePositives)
  def recall(cf: ConfusionMatrix): Double = cf.truePositives.toDouble/(cf.truePositives + cf.falseNegatives)
  def f1(cf: ConfusionMatrix): Double = 2*recall(cf)*precision(cf)/(recall(cf) + precision(cf))

  def getIndividualModelMetrics(df: DataFrame,
                                actualColumnName: String,
                                predictedColumnName: String): IndividualModelMetrics = {
    val cf = evaluate(df, actualColumnName, predictedColumnName)
    IndividualModelMetrics(accuracy = accuracy(cf),
      precision = precision(cf),
      recall = recall(cf),
      f1 = f1(cf))
  }

}
