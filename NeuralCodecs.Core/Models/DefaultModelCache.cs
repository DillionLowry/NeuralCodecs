using NeuralCodecs.Core.Exceptions;
using NeuralCodecs.Core.Interfaces;
using NeuralCodecs.Core.Loading;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Models
{
    // Implementation of DefaultModelCache
    public class DefaultModelCache : IModelCache
    {
        private readonly string _cacheRoot;
        private static readonly SemaphoreSlim _cacheLock = new(1);

        public DefaultModelCache(string? cacheRoot = null)
        {
            _cacheRoot = cacheRoot ?? GetDefaultCacheDirectory();
            Directory.CreateDirectory(_cacheRoot);
        }

        public string GetCacheDirectory() => _cacheRoot;

        public string GetDefaultCacheDirectory()
        {
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".cache", "neural_codecs", "torch");
        }

        public async Task<string?> GetCachedPath(string modelId, string revision)
        {
            var modelDir = GetModelCacheDir(modelId, revision);
            var modelPath = Path.Combine(modelDir, "pytorch_model.bin");
            var configPath = Path.Combine(modelDir, "config.json");
            var metaPath = Path.Combine(modelDir, "cache_meta.json");

            if (File.Exists(modelPath) &&
                File.Exists(configPath) &&
                File.Exists(metaPath))
            {
                // Validate cache metadata
                if (await ValidateCacheMetadata(metaPath))
                {
                    return modelPath;
                }
            }

            return null;
        }

        public async Task<string> CacheModel(string modelId, string sourcePath, string revision)
        {
            var targetDir = GetModelCacheDir(modelId, revision);
            var targetPath = Path.Combine(targetDir, "pytorch_model.bin");

            await _cacheLock.WaitAsync();
            try
            {
                Directory.CreateDirectory(targetDir);

                // Copy model file
                await using (var source = File.OpenRead(sourcePath))
                await using (var target = File.Create(targetPath))
                {
                    await source.CopyToAsync(target);
                }

                // Create metadata
                await CreateCacheMetadata(targetDir, modelId, revision, sourcePath);

                return targetPath;
            }
            finally
            {
                _cacheLock.Release();
            }
        }

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
                throw new ModelLoadException("Failed to clear cache", ex);
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

        private async Task CreateCacheMetadata(string cacheDir, string modelId, string revision, string sourcePath)
        {
            var meta = new CacheMetadata
            {
                ModelId = modelId,
                Revision = revision,
                Timestamp = DateTime.UtcNow,
                MaxAgeInDays = 30,
                Files = new List<CachedFile>
            {
                new()
                {
                    Path = Path.GetFileName(sourcePath),
                    Hash = ComputeFileHash(sourcePath)
                }
            }
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
