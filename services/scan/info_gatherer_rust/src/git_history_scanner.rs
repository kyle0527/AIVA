use git2::{Repository, Oid, Commit, DiffOptions, DiffFormat};
use std::path::Path;
use tracing::{info, warn, error};
use crate::secret_detector::{SecretDetector, SecretFinding};

/// Git 歷史記錄掃描器
pub struct GitHistoryScanner {
    detector: SecretDetector,
    max_commits: usize,
}

/// Git 憑證發現（包含提交資訊）
#[derive(Debug)]
pub struct GitSecretFinding {
    pub finding: SecretFinding,
    pub commit_hash: String,
    pub author: String,
    pub commit_date: String,
    pub commit_message: String,
}

impl GitHistoryScanner {
    /// 建立 Git 歷史掃描器
    pub fn new(max_commits: usize) -> Self {
        Self {
            detector: SecretDetector::new(),
            max_commits,
        }
    }

    /// 掃描 Git 儲存庫
    pub fn scan_repository(&self, repo_path: &Path) -> Result<Vec<GitSecretFinding>, git2::Error> {
        info!("Opening Git repository: {:?}", repo_path);
        
        let repo = Repository::open(repo_path)?;
        let mut findings = Vec::new();

        // 取得所有提交
        let mut revwalk = repo.revwalk()?;
        revwalk.push_head()?;
        revwalk.set_sorting(git2::Sort::TIME)?;

        let mut commit_count = 0;

        for oid_result in revwalk {
            if commit_count >= self.max_commits {
                info!("Reached max commits limit: {}", self.max_commits);
                break;
            }

            let oid = oid_result?;
            let commit = repo.find_commit(oid)?;

            // 掃描提交的差異
            let commit_findings = self.scan_commit(&repo, &commit)?;
            findings.extend(commit_findings);

            commit_count += 1;

            if commit_count % 100 == 0 {
                info!("Scanned {} commits, found {} secrets", commit_count, findings.len());
            }
        }

        info!("Git scan completed: {} commits scanned, {} secrets found", 
              commit_count, findings.len());

        Ok(findings)
    }

    /// 掃描單一提交
    fn scan_commit(&self, repo: &Repository, commit: &Commit) -> Result<Vec<GitSecretFinding>, git2::Error> {
        let mut findings = Vec::new();

        // 取得提交的樹
        let tree = commit.tree()?;
        
        // 取得父提交（如果有）
        let parent_tree = if commit.parent_count() > 0 {
            Some(commit.parent(0)?.tree()?)
        } else {
            None
        };

        // 計算差異
        let mut diff_options = DiffOptions::new();
        let diff = if let Some(parent_tree) = parent_tree {
            repo.diff_tree_to_tree(Some(&parent_tree), Some(&tree), Some(&mut diff_options))?
        } else {
            // 第一個提交，與空樹比較
            repo.diff_tree_to_tree(None, Some(&tree), Some(&mut diff_options))?
        };

        // 提取提交資訊
        let commit_hash = commit.id().to_string();
        let author = commit.author().name().unwrap_or("Unknown").to_string();
        let commit_date = commit.time().seconds().to_string();
        let commit_message = commit.message().unwrap_or("").to_string();

        // 掃描差異內容
        diff.print(DiffFormat::Patch, |_delta, _hunk, line| {
            // 只掃描新增的行（+ 開頭）
            if line.origin() == '+' {
                if let Ok(content) = std::str::from_utf8(line.content()) {
                    let file_path = _delta.new_file().path()
                        .and_then(|p| p.to_str())
                        .unwrap_or("unknown");

                    let secret_findings = self.detector.scan_content(content, file_path);
                    
                    for finding in secret_findings {
                        findings.push(GitSecretFinding {
                            finding,
                            commit_hash: commit_hash.clone(),
                            author: author.clone(),
                            commit_date: commit_date.clone(),
                            commit_message: commit_message.clone(),
                        });
                    }
                }
            }
            true
        })?;

        Ok(findings)
    }

    /// 掃描特定分支
    pub fn scan_branch(&self, repo_path: &Path, branch_name: &str) -> Result<Vec<GitSecretFinding>, git2::Error> {
        let repo = Repository::open(repo_path)?;
        let mut findings = Vec::new();

        // 切換到指定分支
        let branch = repo.find_branch(branch_name, git2::BranchType::Local)?;
        let reference = branch.get();
        let oid = reference.target().ok_or_else(|| {
            git2::Error::from_str("Branch has no target")
        })?;

        // 取得分支的提交歷史
        let mut revwalk = repo.revwalk()?;
        revwalk.push(oid)?;
        revwalk.set_sorting(git2::Sort::TIME)?;

        let mut commit_count = 0;

        for oid_result in revwalk {
            if commit_count >= self.max_commits {
                break;
            }

            let oid = oid_result?;
            let commit = repo.find_commit(oid)?;
            let commit_findings = self.scan_commit(&repo, &commit)?;
            findings.extend(commit_findings);

            commit_count += 1;
        }

        Ok(findings)
    }

    /// 掃描特定檔案的歷史
    pub fn scan_file_history(&self, repo_path: &Path, file_path: &str) -> Result<Vec<GitSecretFinding>, git2::Error> {
        let repo = Repository::open(repo_path)?;
        let mut findings = Vec::new();

        let mut revwalk = repo.revwalk()?;
        revwalk.push_head()?;

        for oid_result in revwalk {
            let oid = oid_result?;
            let commit = repo.find_commit(oid)?;
            let tree = commit.tree()?;

            // 嘗試取得檔案
            if let Ok(entry) = tree.get_path(Path::new(file_path)) {
                if let Ok(object) = entry.to_object(&repo) {
                    if let Some(blob) = object.as_blob() {
                        if let Ok(content) = std::str::from_utf8(blob.content()) {
                            let secret_findings = self.detector.scan_content(content, file_path);

                            for finding in secret_findings {
                                findings.push(GitSecretFinding {
                                    finding,
                                    commit_hash: commit.id().to_string(),
                                    author: commit.author().name().unwrap_or("Unknown").to_string(),
                                    commit_date: commit.time().seconds().to_string(),
                                    commit_message: commit.message().unwrap_or("").to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(findings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_repo() -> (TempDir, Repository) {
        let temp_dir = TempDir::new().unwrap();
        let repo = Repository::init(temp_dir.path()).unwrap();
        
        // 設定 Git 身分
        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test User").unwrap();
        config.set_str("user.email", "test@example.com").unwrap();

        (temp_dir, repo)
    }

    #[test]
    fn test_scan_repository() {
        let (_temp_dir, repo) = create_test_repo();
        
        // 建立包含密鑰的檔案
        let file_path = repo.path().parent().unwrap().join("config.env");
        fs::write(&file_path, "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE").unwrap();

        // 提交檔案
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("config.env")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();

        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "Add config file",
            &tree,
            &[],
        ).unwrap();

        // 執行掃描
        let scanner = GitHistoryScanner::new(100);
        let findings = scanner.scan_repository(repo.path().parent().unwrap()).unwrap();

        // 應該會找到 AWS 密鑰
        assert!(!findings.is_empty());
    }
}
