using NeuralCodecs.Core.Exceptions;
using NeuralCodecs.Core.Utils;
using System.Net.Http.Headers;
using System.Text.Json;

namespace NeuralCodecs.Core.Loading.Repository
{
    internal class GitHubRepositoryInfo
    {
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public GitHubUser Owner { get; set; } = new();
        public List<string>? Topics { get; set; }
        public DateTime UpdatedAt { get; set; }
    }

    /// <summary>
    /// Implements the model repository interface for GitHub.
    /// Provides functionality to download machine learning models from GitHub releases or repositories.
    /// </summary>
    public class GitHubRepository : IModelRepository, IDisposable
    {
        private readonly HttpClient _client;
        private const string API_BASE = "https://api.github.com";
        private const string RAW_BASE = "https://raw.githubusercontent.com";
        private const string RELEASE_BASE = "https://github.com";
        private const int DEFAULT_TIMEOUT_SECONDS = 300;
        private readonly string[] _allowedFilePatterns = new[] { "*.bin", "*.pt", "*.pth", "*.json" };
        private readonly RateLimiter _rateLimiter;
        private bool disposedValue;

        /// <summary>
        /// Initializes a new instance of the GitHubRepository class.
        /// </summary>
        /// <param name="personalAccessToken">Optional GitHub Personal Access Token for authentication.</param>
        public GitHubRepository(string? personalAccessToken = null)
        {
            _client = CreateHttpClient(personalAccessToken);
            _rateLimiter = new RateLimiter(TimeSpan.FromHours(1), 5000); // GitHub's default rate limit
        }

        private HttpClient CreateHttpClient(string? personalAccessToken)
        {
            var client = new HttpClient
            {
                Timeout = TimeSpan.FromSeconds(DEFAULT_TIMEOUT_SECONDS)
            };

            client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/vnd.github+json"));
            client.DefaultRequestHeaders.Add("User-Agent", "NeuralCodecs/1.0");
            client.DefaultRequestHeaders.Add("X-GitHub-Api-Version", "2022-11-28");

            if (!string.IsNullOrEmpty(personalAccessToken))
            {
                client.DefaultRequestHeaders.Authorization =
                    new AuthenticationHeaderValue("Bearer", personalAccessToken);
            }

            return client;
        }

        /// <summary>
        /// Retrieves the path of the main model file from the specified repository or release.
        /// </summary>
        /// <param name="repoId">The GitHub repository identifier (owner/repo).</param>
        /// <param name="revision">The repository tag, branch, or release.</param>
        /// <returns>The path to the main model file within the repository.</returns>
        public async Task<string> GetModelPath(string repoId, string revision)
        {
            try
            {
                await _rateLimiter.WaitForResource();

                // First try to find the file in releases
                var release = await GetRelease(repoId, revision);
                if (release != null)
                {
                    var asset = release.Assets.FirstOrDefault(a =>
                        _allowedFilePatterns.Any(p => new WildcardPattern(p).IsMatch(a.Name)));

                    if (asset != null)
                    {
                        return asset.Name;
                    }
                }

                // Fallback to repository contents
                var files = await GetRepositoryContents(repoId, revision);
                var modelFile = files.FirstOrDefault(f =>
                    _allowedFilePatterns.Any(p => new WildcardPattern(p).IsMatch(f.Path)));

                if (modelFile == null)
                {
                    throw new LoadException($"No model file found in repository: {repoId}");
                }

                return modelFile.Path;
            }
            catch (Exception ex)
            {
                throw new LoadException($"Failed to get model path for {repoId}", ex);
            }
        }

        /// <summary>
        /// Retrieves metadata information about a model from the repository.
        /// </summary>
        /// <param name="repoId">The GitHub repository identifier (owner/repo).</param>
        public async Task<ModelMetadata> GetModelInfo(string repoId)
        {
            try
            {
                await _rateLimiter.WaitForResource();

                var repoInfo = await GetRepositoryInfo(repoId);
                var files = await GetRepositoryContents(repoId, "main");

                var modelFile = files.FirstOrDefault(f =>
                    _allowedFilePatterns.Any(p => new WildcardPattern(p).IsMatch(f.Path)));

                if (modelFile == null)
                {
                    throw new LoadException($"No model file found in repository: {repoId}");
                }

                var configFile = files.FirstOrDefault(f =>
                {
                    var fileName = f.Path.ToLowerInvariant();
                    var modelName = Path.GetFileNameWithoutExtension(modelFile.Path).ToLowerInvariant();

                    return fileName == "config.json" ||
                           fileName == $"{modelName}.json" ||
                           fileName == $"{modelName}_config.json" ||
                           fileName.EndsWith("_config.json") ||
                           (fileName.Contains("config") && fileName.EndsWith(".json"));
                });

                return new ModelMetadata
                {
                    Source = repoId,
                    LastModified = repoInfo.UpdatedAt,
                    Author = repoInfo.Owner.Login,
                    Tags = repoInfo.Topics ?? new List<string>(),
                    Backend = "TorchSharp",
                    Size = modelFile.Size,
                    FileName = modelFile.Path,
                    ConfigFileName = configFile?.Path ?? "",
                    //Description = repoInfo.Description
                };
            }
            catch (Exception ex)
            {
                throw new LoadException($"Failed to get model info for {repoId}", ex);
            }
        }

        /// <summary>
        /// Downloads a model and its associated files from the repository.
        /// </summary>
        /// <param name="repoId">The GitHub repository identifier (owner/repo).</param>
        /// <param name="targetPath">The local directory path where the model should be downloaded.</param>
        /// <param name="progress">An IProgress object to report download progress.</param>
        public async Task DownloadModel(string repoId, string targetPath, IProgress<double> progress, ModelLoadOptions options)
        {
            try
            {
                await _rateLimiter.WaitForResource();

                var files = await GetRepositoryContents(repoId, "main");
                var filesToDownload = files.Where(f =>
                    _allowedFilePatterns.Any(p => new WildcardPattern(p).IsMatch(f.Path))).ToList();

                var totalSize = filesToDownload.Sum(f => f.Size);
                var downloadedSize = 0L;

                foreach (var file in filesToDownload)
                {
                    var destPath = Path.Combine(targetPath, file.Path);
                    var destDir = Path.GetDirectoryName(destPath);
                    if (destDir != null) Directory.CreateDirectory(destDir);

                    // Check if file is stored in LFS
                    var isLfsFile = await IsLfsFile(repoId, file.Path);

                    if (isLfsFile)
                    {
                        await DownloadLfsFile(repoId, file, destPath, new Progress<long>(bytesDownloaded =>
                        {
                            downloadedSize += bytesDownloaded;
                            progress.Report((double)downloadedSize / totalSize);
                        }));
                    }
                    else
                    {
                        await DownloadFile(repoId, file, destPath, new Progress<long>(bytesDownloaded =>
                        {
                            downloadedSize += bytesDownloaded;
                            progress.Report((double)downloadedSize / totalSize);
                        }));
                    }
                }

                await ValidateDownload(targetPath, filesToDownload, options);
            }
            catch (Exception ex)
            {
                // Cleanup on failure
                try
                {
                    if (Directory.Exists(targetPath))
                    {
                        Directory.Delete(targetPath, recursive: true);
                    }
                }
                catch { /* Ignore cleanup errors */ }

                throw new LoadException($"Failed to download model {repoId}", ex);
            }
        }

        private async Task<GitHubRelease?> GetRelease(string repoId, string tag)
        {
            try
            {
                var response = await _client.GetAsync($"{API_BASE}/repos/{repoId}/releases/tags/{tag}");

                if (!response.IsSuccessStatusCode)
                    return null;

                var json = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize<GitHubRelease>(json,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            }
            catch
            {
                return null;
            }
        }

        private async Task<List<GitHubContent>> GetRepositoryContents(string repoId, string revision)
        {
            var contents = new List<GitHubContent>();
            await GetContentsRecursive(repoId, "", revision, contents);
            return contents;
        }

        private async Task GetContentsRecursive(string repoId, string path, string revision, List<GitHubContent> contents)
        {
            await _rateLimiter.WaitForResource();

            var response = await _client.GetAsync($"{API_BASE}/repos/{repoId}/contents/{path}?ref={revision}");
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var items = JsonSerializer.Deserialize<List<GitHubContent>>(json,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            if (items == null)
                return;

            foreach (var item in items)
            {
                if (item.Type == "file")
                {
                    contents.Add(item);
                }
                else if (item.Type == "dir")
                {
                    await GetContentsRecursive(repoId, item.Path, revision, contents);
                }
            }
        }

        private async Task<GitHubRepositoryInfo> GetRepositoryInfo(string repoId)
        {
            await _rateLimiter.WaitForResource();

            var response = await _client.GetAsync($"{API_BASE}/repos/{repoId}");
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<GitHubRepositoryInfo>(json,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
                ?? throw new LoadException("Failed to get repository information");
        }

        private async Task<bool> IsLfsFile(string repoId, string path)
        {
            await _rateLimiter.WaitForResource();

            try
            {
                var response = await _client.GetAsync($"{RAW_BASE}/{repoId}/main/{path}");
                var content = await response.Content.ReadAsStringAsync();

                return content.StartsWith("version https://git-lfs.github.com/spec/");
            }
            catch
            {
                return false;
            }
        }

        private async Task DownloadLfsFile(
            string repoId,
            GitHubContent file,
            string destPath,
            IProgress<long> progress,
            CancellationToken cancellationToken = default)
        {
            // Get LFS pointer
            var response = await _client.GetAsync($"{RAW_BASE}/{repoId}/main/{file.Path}");
            var pointer = await response.Content.ReadAsStringAsync();

            // Parse the OID from the pointer
            var oid = pointer.Split('\n')
                .First(line => line.StartsWith("oid"))
                .Split(':')[1].Trim();

            // Download from LFS store
            var lfsUrl = $"{RELEASE_BASE}/{repoId}.git/info/lfs/objects/batch";
            var lfsRequest = new
            {
                operation = "download",
                objects = new[] { new { oid } }
            };

            var lfsResponse = await _client.PostAsync(lfsUrl,
                new StringContent(JsonSerializer.Serialize(lfsRequest)));
            var lfsData = await lfsResponse.Content.ReadAsStringAsync();
            var lfsResult = JsonSerializer.Deserialize<GitHubLfsResponse>(lfsData);

            if (lfsResult?.Objects?[0].Actions?.Download?.Href == null)
                throw new LoadException("Failed to get LFS download URL");

            await DownloadFileFromUrl(
                lfsResult.Objects[0].Actions.Download.Href,
                destPath,
                progress,
                cancellationToken);
        }

        private async Task DownloadFile(
            string repoId,
            GitHubContent file,
            string destPath,
            IProgress<long> progress,
            CancellationToken cancellationToken = default)
        {
            await DownloadFileFromUrl(
                $"{RAW_BASE}/{repoId}/main/{file.Path}",
                destPath,
                progress,
                cancellationToken);
        }

        private async Task DownloadFileFromUrl(
            string url,
            string destPath,
            IProgress<long> progress,
            CancellationToken cancellationToken = default)
        {
            var tempPath = destPath + ".tmp";

            try
            {
                await _rateLimiter.WaitForResource();

                using var response = await _client.GetAsync(
                    url,
                    HttpCompletionOption.ResponseHeadersRead,
                    cancellationToken);
                response.EnsureSuccessStatusCode();

                await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
                await using var fileStream = new FileStream(
                    tempPath,
                    FileMode.Create,
                    FileAccess.Write,
                    FileShare.None,
                    bufferSize: 81920,
                    useAsync: true);

                await contentStream.CopyToAsync(fileStream, progress, cancellationToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                if (File.Exists(tempPath))
                    File.Delete(tempPath);

                throw new LoadException($"Failed to download file", ex);
            }

            try
            {
                // Atomic move
                if (File.Exists(destPath))
                    File.Delete(destPath);
                File.Move(tempPath, destPath);
            }
            catch (Exception ex)
            {
                throw new LoadException($"Failed to move downloaded file", ex);
            }
        }

        private async Task ValidateDownload(string targetDir, List<GitHubContent> expectedFiles, ModelLoadOptions options)
        {
            // Verify all required files exist
            var missingFiles = expectedFiles.Where(f =>
                !File.Exists(Path.Combine(targetDir, f.Path))).ToList();

            if (missingFiles.Any())
            {
                throw new LoadException(
                    $"Missing files after download: {string.Join(", ",
missingFiles.Select(f => f.Path))}");
            }

            // Verify model file exists
            var modelFile = expectedFiles.FirstOrDefault(f => FileUtil.IsValidModelFile(f.Path));

            if (modelFile == null)
            {
                throw new LoadException("No model file found in downloaded content");
            }

            var modelPath = Path.Combine(targetDir, modelFile.Path);
            if (options.HasConfigFile)
            {
                // Verify config exists
                var configPath = Path.Combine(targetDir, "config.json");
                if (!File.Exists(configPath))
                {
                    configPath = Path.ChangeExtension(modelPath, ".json");
                    if (!File.Exists(configPath))
                    {
                        throw new LoadException("Config file missing from download");
                    }
                }
            }

            // Verify file sizes
            foreach (var file in expectedFiles)
            {
                var filePath = Path.Combine(targetDir, file.Path);
                var fileInfo = new FileInfo(filePath);

                if (fileInfo.Length != file.Size)
                {
                    throw new LoadException(
                        $"File size mismatch for {file.Path}. " +
                        $"Expected {file.Size} bytes but got {fileInfo.Length}");
                }
            }

            // Verify model file format
            if (options.ValidateModel)
            {
                var fileType = await FileUtil.DetectFileTypeFromContentsAsync(modelPath);

                if (fileType is ModelFileType.Unknown)
                {
                    throw new LoadException("Failed to validate model file type");
                }
            }
        }

        private bool IsPyTorchModelFile(byte[] bytes)
        {
            if (bytes.Length < 8) return false;

            // Look for common PyTorch model signatures
            return bytes[0] == 0x80 && bytes[1] == 0x02 || // Pickle protocol
                   bytes[0] == 'P' && bytes[1] == 'K';     // ZIP archive
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    _client.Dispose();
                    _rateLimiter.Dispose();
                }
                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }

    // Supporting classes for JSON deserialization
    internal class GitHubRelease
    {
        public string TagName { get; set; } = "";
        public List<GitHubAsset> Assets { get; set; } = new();
        public string TarballUrl { get; set; } = "";
        public string ZipballUrl { get; set; } = "";
    }

    internal class GitHubAsset
    {
        public string Name { get; set; } = "";
        public string BrowserDownloadUrl { get; set; } = "";
        public long Size { get; set; }
    }

    internal class GitHubContent
    {
        public string Name { get; set; } = "";
        public string Path { get; set; } = "";
        public string Type { get; set; } = "";
        public long Size { get; set; }
        public string DownloadUrl { get; set; } = "";
    }

    internal class GitHubUser
    {
        public string Login { get; set; } = "";
    }

    internal class GitHubLfsResponse
    {
        public List<GitHubLfsObject> Objects { get; set; } = new();
    }

    internal class GitHubLfsObject
    {
        public string Oid { get; set; } = "";
        public GitHubLfsActions Actions { get; set; } = new();
    }

    internal class GitHubLfsActions
    {
        public GitHubLfsDownload? Download { get; set; }
    }

    internal class GitHubLfsDownload
    {
        public string Href { get; set; } = "";
    }

    internal class RateLimiter : IDisposable
    {
        private readonly SemaphoreSlim _semaphore;
        private readonly Queue<DateTime> _requestTimestamps;
        private readonly TimeSpan _timeWindow;
        private readonly int _maxRequests;
        private bool disposedValue;

        public RateLimiter(TimeSpan timeWindow, int maxRequests)
        {
            _timeWindow = timeWindow;
            _maxRequests = maxRequests;
            _semaphore = new SemaphoreSlim(1, 1);
            _requestTimestamps = new Queue<DateTime>();
        }

        public async Task WaitForResource()
        {
            await _semaphore.WaitAsync();
            try
            {
                var now = DateTime.UtcNow;

                // Remove timestamps outside the time window
                while (_requestTimestamps.Count > 0 &&
                       now - _requestTimestamps.Peek() > _timeWindow)
                {
                    _requestTimestamps.Dequeue();
                }

                // If at capacity, wait until oldest request expires
                if (_requestTimestamps.Count >= _maxRequests)
                {
                    var oldestRequest = _requestTimestamps.Peek();
                    var waitTime = _timeWindow - (now - oldestRequest);
                    if (waitTime > TimeSpan.Zero)
                    {
                        await Task.Delay(waitTime);
                    }
                    // Remove expired timestamps again after waiting
                    while (_requestTimestamps.Count > 0 &&
                           DateTime.UtcNow - _requestTimestamps.Peek() > _timeWindow)
                    {
                        _requestTimestamps.Dequeue();
                    }
                }

                _requestTimestamps.Enqueue(DateTime.UtcNow);
            }
            finally
            {
                _semaphore.Release();
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    _semaphore.Dispose();
                }
                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}