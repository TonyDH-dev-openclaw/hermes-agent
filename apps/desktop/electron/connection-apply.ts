async function applyConnectionChange({
  cancelAndWait,
  isPrimary,
  rehomePrimary = null,
  scope,
  sendApplied,
  stopPool,
  teardownPrimary,
  teardownSsh
}) {
  await cancelAndWait(scope)
  await teardownSsh(scope)

  if (!isPrimary) {
    stopPool(scope)

    return
  }

  if (rehomePrimary) {
    await rehomePrimary()

    return
  }

  await teardownPrimary()
  sendApplied()
}

function commitConnectionFailure(current, starting, commit) {
  if (current !== starting) {
    return false
  }

  commit()

  return true
}

// Tony: "Terminal failed to start: ... Remote connection is not ready
// yet. Try again in a moment." Root cause: this checked getTarget()
// exactly once right after ensureBackend()'s single await resolved --
// but ensureBackend() only confirms the underlying backend connection
// process is UNDERWAY, not that whatever separately updates getTarget()
// (the actual SSH tunnel becoming ready) has already landed by that
// exact instant. A real WSL/SSH cold start (right after boot, or after
// this fork's own ssh.service-not-enabled-at-boot fix needed a moment
// to take effect) can easily still be finishing microseconds-to-seconds
// after ensureBackend()'s promise resolves, and the terminal request
// failed outright instead of giving it a genuine moment -- exactly what
// the error's own text tells the user to do manually ("try again"),
// just not automated. Does NOT call ensureBackend() more than once
// (see "does not start a local terminal while configured SSH remains
// unavailable" -- a persistently broken/misconfigured SSH target must
// still fail, not retry forever) -- only re-polls the already-cheap,
// synchronous getTarget() a bounded number of times first.
async function resolveTerminalConnection(getTarget, ensureBackend, delay = ms => new Promise(r => setTimeout(r, ms))) {
  let target = getTarget()

  if (target !== 'pending') {
    return target
  }

  await ensureBackend()
  target = getTarget()

  const POLL_ATTEMPTS = 10
  const POLL_INTERVAL_MS = 500

  for (let attempt = 0; target === 'pending' && attempt < POLL_ATTEMPTS; attempt++) {
    await delay(POLL_INTERVAL_MS)
    target = getTarget()
  }

  if (target === 'pending') {
    throw new Error('Remote connection is not ready yet. Try again in a moment.')
  }

  return target
}

export { applyConnectionChange, commitConnectionFailure, resolveTerminalConnection }
