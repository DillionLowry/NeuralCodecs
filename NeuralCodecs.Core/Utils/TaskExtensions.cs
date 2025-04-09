using Microsoft.Extensions.Logging;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralCodecs.Core.Utils
{
    /// <summary>
    /// Provides extension methods for Task-based operations.
    /// </summary>
    internal static class TaskExtensions
    {
        /// <summary>
        /// Executes multiple tasks with a degree of parallelism
        /// </summary>
        public static async Task ForEachAsync<T>(
            this IEnumerable<T> source,
            Func<T, Task> operation,
            int maxDegreeOfParallelism = 4,
            CancellationToken cancellationToken = default)
        {
            var tasks = new List<Task>();
            var throttler = new SemaphoreSlim(maxDegreeOfParallelism);

            foreach (var item in source)
            {
                await throttler.WaitAsync(cancellationToken).ConfigureAwait(false);

                tasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        await operation(item).ConfigureAwait(false);
                    }
                    finally
                    {
                        throttler.Release();
                    }
                }, cancellationToken));
            }

            await Task.WhenAll(tasks).ConfigureAwait(false);
        }

        /// <summary>
        /// Executes an operation with automatic retry on failure.
        /// </summary>
        /// <typeparam name="T">The type of the operation result.</typeparam>
        /// <param name="operation">The operation to execute.</param>
        /// <param name="maxAttempts">Maximum number of retry attempts.</param>
        /// <param name="delay">Delay between retry attempts.</param>
        /// <param name="ct">Optional cancellation token.</param>
        /// <returns>The result of the operation if successful.</returns>
        /// <exception cref="AggregateException">Thrown when all retry attempts fail.</exception>
        public static async ValueTask<T> WithRetry<T>(
            this Func<CancellationToken, ValueTask<T>> operation,
            int maxAttempts = 3,
            TimeSpan? delay = null,
            CancellationToken ct = default)
        {
            delay ??= TimeSpan.FromSeconds(1);
            Exception lastException = null;

            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                try
                {
                    return await operation(ct).ConfigureAwait(false);
                }
                catch (Exception ex) when (attempt < maxAttempts - 1 &&
                                        ex is not OperationCanceledException)
                {
                    lastException = ex;
                    await Task.Delay(delay.Value, ct).ConfigureAwait(false);
                }
            }

            throw new AggregateException($"Operation failed after {maxAttempts} attempts", lastException);
        }

        /// <summary>
        /// Executes an operation with automatic retry
        /// </summary>
        public static async Task<T> WithRetry<T>(
            this Func<CancellationToken, Task<T>> operation,
            int maxAttempts = 3,
            Func<int, Exception, TimeSpan> retryDelay = null,
            ILogger logger = null,
            CancellationToken cancellationToken = default,
            [CallerMemberName] string callerName = null)
        {
            retryDelay ??= (attempt, ex) => TimeSpan.FromSeconds(Math.Pow(2, attempt - 1)); // Exponential backoff

            Exception lastException = null;

            for (int attempt = 1; attempt <= maxAttempts; attempt++)
            {
                try
                {
                    if (attempt > 1)
                    {
                        logger?.LogDebug("Retry attempt {Attempt}/{MaxAttempts} for {Operation}",
                            attempt, maxAttempts, callerName);
                    }

                    return await operation(cancellationToken).ConfigureAwait(false);
                }
                catch (Exception ex) when (attempt < maxAttempts &&
                                           ex is not OperationCanceledException &&
                                           !cancellationToken.IsCancellationRequested)
                {
                    lastException = ex;
                    var delay = retryDelay(attempt, ex);

                    logger?.LogWarning(ex, "Operation {Operation} failed (attempt {Attempt}/{MaxAttempts}), retrying in {Delay}ms",
                        callerName, attempt, maxAttempts, delay.TotalMilliseconds);

                    await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
                }
            }

            // If we get here, all attempts failed
            logger?.LogError(lastException, "Operation {Operation} failed after {MaxAttempts} attempts",
                callerName, maxAttempts);

            throw new AggregateException($"Operation failed after {maxAttempts} attempts", lastException);
        }

        /// <summary>
        /// Executes an operation with automatic retry
        /// </summary>
        public static async ValueTask<T> WithRetry<T>(
            this Func<CancellationToken, ValueTask<T>> operation,
            int maxAttempts = 3,
            Func<int, Exception, TimeSpan> retryDelay = null,
            ILogger logger = null,
            CancellationToken cancellationToken = default,
            [CallerMemberName] string callerName = null)
        {
            retryDelay ??= (attempt, ex) => TimeSpan.FromSeconds(Math.Pow(2, attempt - 1)); // Exponential backoff

            Exception lastException = null;

            for (int attempt = 1; attempt <= maxAttempts; attempt++)
            {
                try
                {
                    if (attempt > 1)
                    {
                        logger?.LogDebug("Retry attempt {Attempt}/{MaxAttempts} for {Operation}",
                            attempt, maxAttempts, callerName);
                    }

                    return await operation(cancellationToken).ConfigureAwait(false);
                }
                catch (Exception ex) when (attempt < maxAttempts &&
                                           ex is not OperationCanceledException &&
                                           !cancellationToken.IsCancellationRequested)
                {
                    lastException = ex;
                    var delay = retryDelay(attempt, ex);

                    logger?.LogWarning(ex, "Operation {Operation} failed (attempt {Attempt}/{MaxAttempts}), retrying in {Delay}ms",
                        callerName, attempt, maxAttempts, delay.TotalMilliseconds);

                    await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
                }
            }

            // If we get here, all attempts failed
            logger?.LogError(lastException, "Operation {Operation} failed after {MaxAttempts} attempts",
                callerName, maxAttempts);

            throw new AggregateException($"Operation failed after {maxAttempts} attempts", lastException);
        }

        /// <summary>
        /// Applies a timeout to the execution of a ValueTask.
        /// </summary>
        /// <typeparam name="T">The type of the task result.</typeparam>
        /// <param name="task">The task to apply timeout to.</param>
        /// <param name="timeout">The maximum time allowed for the operation.</param>
        /// <param name="operationName">Name of the operation for error reporting.</param>
        /// <param name="ct">Optional cancellation token.</param>
        /// <returns>The result of the task if completed within the timeout period.</returns>
        /// <exception cref="TimeoutException">Thrown when the operation exceeds the specified timeout.</exception>
        public static async ValueTask<T> WithTimeout<T>(
                  this ValueTask<T> task,
                  TimeSpan timeout,
                  string operationName = "operation",
                  CancellationToken cancellationToken = default)
        {
            // Create a linked cancellation token source that will be canceled
            // either when the original token is canceled or when the timeout expires
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(timeout);

            try
            {
                if (task.IsCompleted)
                {
                    return await task.ConfigureAwait(false);
                }

                // Wait for the task or for the timeout to expire
                return await task.AsTask().WaitAsync(timeoutCts.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
            {
                // If the operation was canceled by our timeout token but not by the user's token,
                // then it's a timeout
                throw new TimeoutException($"The {operationName} timed out after {timeout.TotalSeconds:F1} seconds");
            }
        }

        /// <summary>
        /// Adds a timeout to a Task
        /// </summary>
        public static async Task<T> WithTimeout<T>(
            this Task<T> task,
            TimeSpan timeout,
            string operationName = "operation",
            CancellationToken cancellationToken = default)
        {
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(timeout);

            try
            {
                return await task.WaitAsync(timeoutCts.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
            {
                throw new TimeoutException($"The {operationName} timed out after {timeout.TotalSeconds:F1} seconds");
            }
        }

        /// <summary>
        /// Times the execution of a task and returns the result and elapsed time
        /// </summary>
        public static async Task<(T Result, TimeSpan Elapsed)> WithTiming<T>(
            this Task<T> task,
            ILogger logger = null,
            string operationName = "operation")
        {
            var sw = Stopwatch.StartNew();

            try
            {
                var result = await task.ConfigureAwait(false);
                sw.Stop();

                logger?.LogDebug("{Operation} completed in {ElapsedMs}ms",
                    operationName, sw.ElapsedMilliseconds);

                return (result, sw.Elapsed);
            }
            catch (Exception ex)
            {
                sw.Stop();
                logger?.LogWarning(ex, "{Operation} failed after {ElapsedMs}ms",
                    operationName, sw.ElapsedMilliseconds);
                throw;
            }
        }
    }
}