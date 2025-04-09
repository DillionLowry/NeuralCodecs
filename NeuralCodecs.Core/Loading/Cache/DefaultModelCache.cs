using NeuralCodecs.Core.Exceptions;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace NeuralCodecs.Core.Loading.Cache
{
    /// <summary>
    /// Default implementation of the IModelCache interface for managing neural network model caching.
    /// Manages the local storage and retrieval of downloaded models with validation and cleanup capabilities.
    /// </summary>
    public class DefaultModelCache : IModelCache
    {
        private readonly string _cacheRoot;
        private static readonly SemaphoreSlim _cacheLock = new(1);

        /// <summary>
        /// Initializes a new instance of the DefaultModelCache class.
        /// </summary>
        /// <param name="cacheRoot">Optional custom cache directory path. If not specified, uses the default cache location.</param>
        public DefaultModelCache(string? cacheRoot = null)
        {
            _cacheRoot = cacheRoot ?? GetDefaultCacheDirectory();
            Directory.CreateDirectory(_cacheRoot);
        }

        /// <summary>
        /// Gets the current cache directory path.
        /// </summary>
        /// <returns>The absolute path to the cache directory.</returns>
        public string GetCacheDirectory() => _cacheRoot;

        /// <summary>
        /// Gets the default cache directory path in the user's home folder.
        /// </summary>
        /// <returns>The default cache directory path.</returns>
        public string GetDefaultCacheDirectory()
        {
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".cache", "neural_codecs", "torch");
        }

        /// <summary>
        /// Attempts to retrieve a cached model path for the specified model ID and revision.
        /// </summary>
        /// <param name="modelId">The unique identifier of the model.</param>
        /// <param name="revision">The specific revision of the model.</param>
        /// <returns>The path to the cached model if found and valid; otherwise, null.</returns>
        public async Task<string?> GetCachedPath(string modelId, string revision)
        {
            var modelDir = GetModelCacheDir(modelId, revision);
            var metaPath = Path.Combine(modelDir, "cache_meta.json");

            if (File.Exists(metaPath) && await GetCacheMetadata(metaPath) is CacheMetadata meta)
            {
                return Path.Combine(modelDir, meta.ModelFileName);
            }

            return null;
        }

        /// <summary>
        /// Caches a model and its configuration files in the local cache directory.
        /// </summary>
        /// <param name="modelId">The unique identifier of the model.</param>
        /// <param name="sourcePath">The source directory containing the model files.</param>
        /// <param name="revision">The specific revision of the model.</param>
        /// <param name="targetFileName">The name of the model file to cache.</param>
        /// <param name="targetConfigFileName">The name of the configuration file to cache.</param>
        /// <param name="additionalMetadata">Optional additional metadata to store with the cached model.</param>
        /// <returns>The path to the cached model file.</returns>
        /// <exception cref="CacheException">Thrown when caching operations fail.</exception>
        /// <exception cref="FileNotFoundException">Thrown when source files are not found.</exception>
        public async Task<string> CacheModel(
            string modelId,
            string sourcePath,
            string revision,
            string targetFileName,
            string targetConfigFileName,
            IDictionary<string, string>? additionalMetadata = null)
        {
            var targetDir = GetModelCacheDir(modelId, revision);
            var targetPath = Path.Combine(targetDir, targetFileName);

            //await _cacheLock.WaitAsync();
            try
            {

                var dir = new DirectoryInfo(sourcePath);
                if (!dir.Exists)
                {
                    throw new DirectoryNotFoundException($"Source directory does not exist: {sourcePath}");
                }

                // Check if we have read access to the directory (only on Windows)
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    var acl = dir.GetAccessControl();
                    var rules = acl.GetAccessRules(true, true, typeof(System.Security.Principal.SecurityIdentifier));
                }

                // Verify source file exists and is accessible
                if (!File.Exists(Path.Combine(sourcePath, targetFileName)))
                {
                    throw new FileNotFoundException($"Source file not found: {sourcePath}");
                }

                if (!Directory.Exists(targetDir))
                {
                    Directory.CreateDirectory(targetDir);
                }

                await using (var source = File.OpenRead(Path.Combine(sourcePath, targetFileName)))
                await using (var target = File.OpenWrite(targetPath))
                {
                    await source.CopyToAsync(target);
                }

                if (!string.IsNullOrWhiteSpace(targetConfigFileName))
                {
                    var targetConfigPath = Path.Combine(targetDir, targetConfigFileName);

                    await using var source = File.OpenRead(Path.Combine(sourcePath, targetConfigFileName));
                    await using var targetConfig = File.OpenWrite(targetConfigPath);
                    await source.CopyToAsync(targetConfig);
                }

                await CreateCacheMetadata(targetDir, modelId, revision, targetFileName, targetConfigFileName);

                return targetPath;
            }
            catch (UnauthorizedAccessException uae)
            {
                throw new CacheException($"Permission denied accessing source directory: {sourcePath}. Please check directory permissions.", uae);
            }
            catch (Exception ex)
            {
                // Cleanup on failure
                if (File.Exists(targetPath))
                {
                    File.Delete(targetPath);
                }
                throw new CacheException($"Failed to cache model file: {targetFileName}", ex);
            }
            finally
            {
                //_cacheLock.Release();
            }
        }

        /// <summary>
        /// Clears the model cache, either completely or for a specific model.
        /// </summary>
        /// <param name="modelId">Optional model ID to clear specific model cache. If null, clears entire cache.</param>
        /// <exception cref="LoadException">Thrown when cache clearing operations fail.</exception>
        public void ClearCache(string? modelId = null)
        {
            try
            {
                if (modelId == null)
                {
                    // Clear all cache
                    if (Directory.Exists(_cacheRoot))
                    {
                        Directory.Delete(_cacheRoot, recursive: true);
                        Directory.CreateDirectory(_cacheRoot);
                    }
                }
                else
                {
                    // Clear specific model
                    var modelDir = Path.Combine(_cacheRoot, modelId.Replace('/', '_'));
                    if (Directory.Exists(modelDir))
                    {
                        Directory.Delete(modelDir, recursive: true);
                    }
                }
            }
            catch (Exception ex)
            {
                throw new LoadException("Failed to clear cache", ex);
            }
        }

        private string GetModelCacheDir(string modelId, string revision)
        {
            var safeModelId = modelId.Replace('/', '_');
            return Path.Combine(_cacheRoot, safeModelId, revision);
        }

        private async Task<bool> ValidateCacheMetadata(string metaPath)
        {
            try
            {
                var meta = JsonSerializer.Deserialize<CacheMetadata>(
                    await File.ReadAllTextAsync(metaPath));

                if (meta == null) return false;

                // Check timestamp isn't too old (e.g., 30 days)
                var ageInDays = (DateTime.UtcNow - meta.Timestamp).TotalDays;
                if (ageInDays > meta.MaxAgeInDays) return false;

                // Verify file hashes
                foreach (var file in meta.Files)
                {
                    var path = Path.Combine(Path.GetDirectoryName(metaPath)!, file.Path);
                    if (!File.Exists(path)) return false;

                    var hash = ComputeFileHash(path);
                    if (hash != file.Hash) return false;
                }

                return true;
            }
            catch
            {
                return false;
            }
        }
        private async Task<CacheMetadata?> GetCacheMetadata(string metaPath)
        {
            try
            {
                var meta = JsonSerializer.Deserialize<CacheMetadata>(
                    await File.ReadAllTextAsync(metaPath));

                if (meta == null) return null;

                // Check timestamp isn't too old (e.g., 30 days)
                var ageInDays = (DateTime.UtcNow - meta.Timestamp).TotalDays;
                if (ageInDays > meta.MaxAgeInDays) return null;

                // Verify file hashes
                foreach (var file in meta.Files)
                {
                    var path = Path.Combine(Path.GetDirectoryName(metaPath)!, file.Path);
                    if (!File.Exists(path))
                    {
                        Console.WriteLine($"File not found: {path}");
                        return null;
                    }
                }

                return meta;
            }
            catch
            {
                Console.WriteLine($"Failed to read cache metadata: {metaPath}");
                return null;
            }
        }
        private async Task CreateCacheMetadata(string cacheDir, string modelId, string revision, string filename, string configFilename = "config.json")
        {
            var fileNames = Directory.GetFiles(cacheDir);
            var meta = new CacheMetadata
            {
                ModelId = modelId,
                Revision = revision,
                Timestamp = DateTime.UtcNow,
                MaxAgeInDays = 30,
                ModelFileName = filename,
                ConfigFileName = configFilename,
                Files = fileNames.Select(fileName => new CachedFile
                {
                    Path = fileName
                }).ToList()
            };

            var metaPath = Path.Combine(cacheDir, "cache_meta.json");
            await File.WriteAllTextAsync(
                metaPath,
                JsonSerializer.Serialize(meta, new JsonSerializerOptions
                {
                    WriteIndented = true
                }));
        }

        private string ComputeFileHash(string path)
        {
            using var sha256 = System.Security.Cryptography.SHA256.Create();
            using var stream = File.OpenRead(path);
            var hash = sha256.ComputeHash(stream);
            return Convert.ToBase64String(hash);
        }
    }
}