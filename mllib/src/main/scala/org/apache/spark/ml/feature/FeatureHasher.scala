/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.feature

import org.apache.spark.annotation.Since
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.mllib.feature.{HashingTF => OldHashingTF}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.util.Utils
import org.apache.spark.util.collection.OpenHashMap

/**
 * Feature hashing projects a set of categorical or numerical features into a feature vector of
 * specified dimension (typically substantially smaller than that of the original feature
 * space). This is done using the hashing trick (https://en.wikipedia.org/wiki/Feature_hashing)
 * to map features to indices in the feature vector.
 *
 * The [[FeatureHasher]] transformer operates on multiple columns. Each column may be numeric
 * (representing a real feature) or string (representing a categorical feature). Boolean columns
 * are also supported, and treated as categorical features. For numeric features, the hash value of
 * the column name is used to map the feature value to its index in the feature vector.
 * For categorical features, the hash value of the string "column_name=value" is used to map to the
 * vector index, with an indicator value of `1.0`. Thus, categorical features are "one-hot" encoded
 * (similarly to using [[OneHotEncoder]] with `dropLast=false`).
 *
 * Null (missing) values are ignored (implicitly zero in the resulting feature vector).
 *
 * Since a simple modulo is used to transform the hash function to a vector index,
 * it is advisable to use a power of two as the numFeatures parameter;
 * otherwise the features will not be mapped evenly to the vector indices.
 *
 * {{{
 *   val df = Seq(
 *    (2.0, true, "1", "foo"),
 *    (3.0, false, "2", "bar")
 *   ).toDF("real", "bool", "stringNum", "string")
 *
 *   val hasher = new FeatureHasher()
 *    .setInputCols("real", "bool", "stringNum", "num")
 *    .setOutputCol("features")
 *
 *   hasher.transform(df).show()
 *
 *   +----+-----+---------+------+--------------------+
 *   |real| bool|stringNum|string|            features|
 *   +----+-----+---------+------+--------------------+
 *   | 2.0| true|        1|   foo|(262144,[51871,63...|
 *   | 3.0|false|        2|   bar|(262144,[6031,806...|
 *   +----+-----+---------+------+--------------------+
 * }}}
 */
@Since("2.3.0")
class FeatureHasher(@Since("2.3.0") override val uid: String) extends Transformer
  with HasInputCols with HasOutputCol with DefaultParamsWritable {

  @Since("2.3.0")
  def this() = this(Identifiable.randomUID("featureHasher"))

  /**
   * Number of features. Should be greater than 0.
   * (default = 2^18^)
   * @group param
   */
  @Since("2.3.0")
  val numFeatures = new IntParam(this, "numFeatures", "number of features (> 0)",
    ParamValidators.gt(0))

  setDefault(numFeatures -> (1 << 18))

  /** @group getParam */
  @Since("2.3.0")
  def getNumFeatures: Int = $(numFeatures)

  /** @group setParam */
  @Since("2.3.0")
  def setNumFeatures(value: Int): this.type = set(numFeatures, value)

  /** @group setParam */
  @Since("2.3.0")
  def setInputCols(values: String*): this.type = setInputCols(values.toArray)

  /** @group setParam */
  @Since("2.3.0")
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** @group setParam */
  @Since("2.3.0")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val hashFunc: Any => Int = OldHashingTF.murmur3Hash
    val n = $(numFeatures)

    val outputSchema = transformSchema(dataset.schema)
    val realFields = outputSchema.fields.filter { f =>
      f.dataType.isInstanceOf[NumericType]
    }.map(_.name).toSet

    def getDouble(x: Any): Double = {
      x match {
        case n: java.lang.Number =>
          n.doubleValue()
        case other =>
          // will throw ClassCastException if it cannot be cast, as would row.getDouble
          other.asInstanceOf[Double]
      }
    }

    val hashFeatures = udf { row: Row =>
      val map = new OpenHashMap[Int, Double]()
      $(inputCols).foreach { case colName =>
        val fieldIndex = row.fieldIndex(colName)
        if (!row.isNullAt(fieldIndex)) {
          val (rawIdx, value) = if (realFields(colName)) {
            // numeric values are kept as is, with vector index based on hash of "column_name"
            val value = getDouble(row.get(fieldIndex))
            val hash = hashFunc(colName)
            (hash, value)
          } else {
            // string and boolean values are treated as categorical, with an indicator value of 1.0
            // and vector index based on hash of "column_name=value"
            val value = row.get(fieldIndex).toString
            val fieldName = s"$colName=$value"
            val hash = hashFunc(fieldName)
            (hash, 1.0)
          }
          val idx = Utils.nonNegativeMod(rawIdx, n)
          map.changeValue(idx, value, v => v + value)
        }
      }
      Vectors.sparse(n, map.toSeq)
    }

    val metadata = outputSchema($(outputCol)).metadata
    dataset.select(
      col("*"),
      hashFeatures(struct($(inputCols).map(col(_)): _*)).as($(outputCol), metadata))
  }

  override def copy(extra: ParamMap): FeatureHasher = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val fields = schema($(inputCols).toSet)
    fields.foreach { case fieldSchema =>
      val dataType = fieldSchema.dataType
      val fieldName = fieldSchema.name
      require(dataType.isInstanceOf[NumericType] ||
        dataType.isInstanceOf[StringType] ||
        dataType.isInstanceOf[BooleanType],
        s"FeatureHasher requires columns to be of NumericType, BooleanType or StringType. " +
          s"Column $fieldName was $dataType")
    }
    val attrGroup = new AttributeGroup($(outputCol), $(numFeatures))
    SchemaUtils.appendColumn(schema, attrGroup.toStructField())
  }
}

@Since("2.3.0")
object FeatureHasher extends DefaultParamsReadable[FeatureHasher] {

  @Since("2.3.0")
  override def load(path: String): FeatureHasher = super.load(path)
}
