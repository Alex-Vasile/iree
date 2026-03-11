// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for semaphore failure semantics:
//   - First failure wins (second fail() must not overwrite)

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class SemaphoreFailureTest : public CtsTestBase<> {};

// Tests that the first failure status wins — a second fail() must not overwrite.
TEST_P(SemaphoreFailureTest, FirstFailureWins) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore();

  // First failure.
  iree_hal_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_DATA_LOSS, "first failure"));

  // Second failure — should be dropped (first-failure-wins).
  iree_hal_semaphore_fail(
      semaphore,
      iree_make_status(IREE_STATUS_PERMISSION_DENIED, "second failure"));

  // Query must return the first failure status.
  uint64_t value = 0;
  iree_status_t query_status = iree_hal_semaphore_query(semaphore, &value);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, query_status);

  iree_hal_semaphore_release(semaphore);
}

CTS_REGISTER_TEST_SUITE(SemaphoreFailureTest);

}  // namespace iree::hal::cts
