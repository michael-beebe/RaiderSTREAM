/**
 * @file RSOpts.cpp
 * @brief Implementation of RaiderSTREAM results
 * @copyright Copyright (C) 2022-2024 Texas Tech University. All Rights Reserved.
 * @author michael.beebe@ttu.edu
 * 
 * See LICENSE in the top level directory for licensing details
 */

#include "RaiderSTREAM/RSRes.h"

/**
 * @brief Constructor for RSRes Object
 * 
 * Allocates data for Result Arrays
 */
RSRes::RSRes(){
  MBPS = CALLOC_DATA(NUM_KERNELS, 1, double)
  FLOPS = CALLOC_DATA(NUM_KERNELS, 1, double)
  TIMES = CALLOC_DATA(NUM_KERNELS, 1, double)
};

/**
 * @brief Destructor for RSRes Object
 * 
 * Frees data for Result Arrays
 */
RSRes::~RSRes() {
  FREE_DATA(MBPS)
  FREE_DATA(FLOPS)
  FREE_DATA(TIMES)
}