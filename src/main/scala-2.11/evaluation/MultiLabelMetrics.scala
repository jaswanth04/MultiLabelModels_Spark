package evaluation

case class MultiLabelMetrics(hammingLoss: Double,
                             subsetLoss: Double,
                             accuracy: Double,
                             precision: Double,
                             recall: Double,
                             f1: Double)
