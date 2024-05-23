  // virtual bool checkErrors(
  //   const char *label,
  //   double *array,
  //   double avgErr,
  //   double expVal,
  //   double epsilon,
  //   int *errors,
  //   ssize_t streamArraySize) = 0;

  // /**
  //  * @brief Check for errors in central benchmark results.
  //  *
  //  * This pure virtual method checks for errors in the central benchmark results.
  //  *
  //  * @param label Label or description for the check.
  //  * @param array Pointer to the array to check.
  //  * @param avgErr Average error threshold.
  //  * @param expVal Expected value.
  //  * @param epsilon Epsilon value for comparison.
  //  * @param errors Pointer to store the number of errors found.
  //  * @param streamArraySize Size of the STREAM array.
  //  * @return True if no errors are found, false otherwise.
  //  */
  // virtual bool centralCheckErrors(
  //   const char *label,
  //   double *array,
  //   double avgErr,
  //   double expVal,
  //   double epsilon,
  //   int *errors,
  //   ssize_t streamArraySize) = 0;

  // /**
  //  * @brief Compute standard errors for the benchmark.
  //  *
  //  * This pure virtual method computes standard errors for the benchmark.
  //  *
  //  * @param aj Index value for array A.
  //  * @param bj Index value for array B.
  //  * @param cj Index value for array C.
  //  * @param streamArraySize Size of the STREAM array.
  //  * @param a Pointer to array A.
  //  * @param b Pointer to array B.
  //  * @param c Pointer to array C.
  //  * @param aSumErr Pointer to store the sum of errors for array A.
  //  * @param bSumErr Pointer to store the sum of errors for array B.
  //  * @param cSumErr Pointer to store the sum of errors for array C.
  //  * @param aAvgErr Pointer to store the average error for array A.
  //  * @param bAvgErr Pointer to store the average error for array B.
  //  * @param cAvgErr Pointer to store the average error for array C.
  //  * @return True if computation is successful, false otherwise.
  //  */
  // virtual bool standardErrors(
  //   double aj, double bj, double cj,
  //   ssize_t streamArraySize,
  //   double *a, double *b, double *c,
  //   double *aSumErr, double *bSumErr, double *cSumErr,
  //   double *aAvgErr, double *bAvgErr, double *cAvgErr) = 0;

  // /**
  //  * @brief Validate values in the benchmark arrays.
  //  *
  //  * This pure virtual method validates values in the benchmark arrays.
  //  *
  //  * @param aj Index value for array A.
  //  * @param bj Index value for array B.
  //  * @param cj Index value for array C.
  //  * @param streamArraySize Size of the STREAM array.
  //  * @param a Pointer to array A.
  //  * @param b Pointer to array B.
  //  * @param c Pointer to array C.
  //  * @return True if values are valid, false otherwise.
  //  */
  // virtual bool validateValues(
  //   double aj, double bj, double cj,
  //   ssize_t streamArraySize,
  //   double *a, double *b, double *c) = 0;

  // /**
  //  * @brief Compute central errors for the benchmark.
  //  *
  //  * This pure virtual method computes central errors for the benchmark.
  //  *
  //  * @param aj Index value for array A.
  //  * @param bj Index value for array B.
  //  * @param cj Index value for array C.
  //  * @param streamArraySize Size of the STREAM array.
  //  * @param a Pointer to array A.
  //  * @param b Pointer to array B.
  //  * @param c Pointer to array C.
  //  * @param aSumErr Pointer to store the sum of errors for array A.
  //  * @param bSumErr Pointer to store the sum of errors for array B.
  //  * @param cSumErr Pointer to store the sum of errors for array C.
  //  * @param aAvgErr Pointer to store the average error for array A.
  //  * @param bAvgErr Pointer to store the average error for array B.
  //  * @param cAvgErr Pointer to store the average error for array C.
  //  * @return True if computation is successful, false otherwise.
  //  */
  // virtual bool centralErrors(
  //   double aj, double bj, double cj,
  //   ssize_t streamArraySize,
  //   double *a, double *b, double *c,
  //   double *aSumErr, double *bSumErr, double *cSumErr,
  //   double *aAvgErr, double *bAvgErr, double *cAvgErr) = 0;

  // /**
  //  * @brief Perform sequential validation.
  //  *
  //  * This pure virtual method performs sequential validation for the benchmark.
  //  *
  //  * @param streamArraySize Size of the STREAM array.
  //  * @param scalar Scalar value.
  //  * @param isValidated Pointer to store the validation result.
  //  * @param a Pointer to array A.
  //  * @param b Pointer to array B.
  //  * @param c Pointer to array C.
  //  * @return True if validation is successful, false otherwise.
  //  */
  // virtual bool seqValidation(
  //   ssize_t streamArraySize,
  //   double scalar,
  //   int *isValidated,
  //   double *a, double *b, double *c) = 0;

  // /**
  //  * @brief Perform gather validation.
  //  *
  //  * This pure virtual method performs gather validation for the benchmark.
  //  *
  //  * @param streamArraySize Size of the STREAM array.
  //  * @param scalar Scalar value.
  //  * @param isValidated Pointer to store the validation result.
  //  * @param a Pointer to array A.
  //  * @param b Pointer to array B.
  //  * @param c Pointer to array C.
  //  * @return True if validation is successful, false otherwise.
  //  */
  // virtual bool gatherValidation(
  //   ssize_t streamArraySize,
  //   double scalar,
  //   int *isValidated,
  //   double *a, double *b, double *c) = 0;

  // /**
  //  * @brief Perform scatter validation.
  //  *
  //  * This pure virtual method performs scatter validation for the benchmark.
  //  *
  //  * @param streamArraySize Size of the STREAM array.
  //  * @param scalar Scalar value.
  //  * @param isValidated Pointer to store the validation result.
  //  * @param a Pointer to array A.
  //  * @param b Pointer to array B.
  //  * @param c Pointer to array C.
  //  * @return True if validation is successful, false otherwise.
  //  */
  // virtual bool scatterValidation(
  //   ssize_t streamArraySize,
  //   double scalar,
  //   int *isValidated,
  //   double *a, double *b, double *c) = 0;

  // /**
  //  * @brief Perform scatter-gather validation.
  //  *
  //  * This pure virtual method performs scatter-gather validation for the benchmark.
  //  *
  //  * @param streamArraySize Size of the STREAM array.
  //  * @param scalar Scalar value.
  //  * @param isValidated Pointer to store the validation result.
  //  * @param a Pointer to array A.
  //  * @param b Pointer to array B.
  //  * @param c Pointer to array C.
  //  * @return True if validation is successful, false otherwise.
  //  */
  // virtual bool sgValidation(
  //   ssize_t streamArraySize,
  //   double scalar,
  //   int *isValidated,
  //   double *a, double *b, double *c) = 0;

  // /**
  //  * @brief Perform central validation.
  //  *
  //  * This pure virtual method performs central validation for the benchmark.
  //  *
  //  * @param streamArraySize Size of the STREAM array.
  //  * @param scalar Scalar value.
  //  * @param isValidated Pointer to store the validation result.
  //  * @param a Pointer to array A.
  //  * @param b Pointer to array B.
  //  * @param c Pointer to array C.
  //  * @return True if validation is successful, false otherwise.
  //  */
  // virtual bool centralValidation(
  //   ssize_t streamArraySize,
  //   double scalar,
  //   int *isValidated,
  //   double *a, double *b, double *c) = 0;

  // /**
  //  * @brief Check the results of the STREAM benchmark.
  //  *
  //  * This pure virtual method checks the results of the STREAM benchmark.
  //  *
  //  * @param isValidated Pointer to store the validation result.
  //  * @return True if validation is successful, false otherwise.
  //  */
  // virtual bool checkSTREAMResults(int *isValidated) = 0;