# Fix Go Engine Dependencies - Remove external logging libraries
# Follow aiva_common standards: use standard library only

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fix Go Engine Dependencies" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$rootPath = "C:\D\fold7\AIVA-git\services\scan\go_engine"

# Files that use zap logger
$zapFiles = @(
    "$rootPath\cmd\ssrf-scanner\main.go",
    "$rootPath\cmd\sca-scanner\main.go",
    "$rootPath\cmd\cspm-scanner\main.go",
    "$rootPath\internal\sca\scanner\sca_scanner.go",
    "$rootPath\internal\cspm\scanner\cspm_scanner.go"
)

# Files that use logrus
$logrusFiles = @(
    "$rootPath\internal\ssrf\verifier\verifier.go",
    "$rootPath\internal\ssrf\oob\monitor.go",
    "$rootPath\internal\sca\fs\walker.go",
    "$rootPath\internal\cspm\audit\aws.go"
)

Write-Host "`nRemoving zap logger from $($zapFiles.Count) files..." -ForegroundColor Yellow

foreach ($file in $zapFiles) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        
        # Replace zap import with log
        $content = $content -replace '"go\.uber\.org/zap"', '"log"'
        
        # Remove zap logger initialization
        $content = $content -replace 'logger,\s*err\s*:=\s*zap\.NewDevelopment\(\)[^}]+}', '// Using standard log package'
        $content = $content -replace 'defer\s+logger\.Sync\(\)', '// Using standard log package'
        
        # Replace logger parameter in NewXXX functions
        $content = $content -replace 'detector\.NewSSRFDetector\(logger\)', 'detector.NewSSRFDetector()'
        $content = $content -replace 'scanner\.NewSCAScanner\(logger\)', 'scanner.NewSCAScanner()'
        $content = $content -replace 'NewWorkerPool\([^,]+,\s*logger\)', 'NewWorkerPool($1)'
        
        # Replace all logger.Info/Warn/Error calls
        $content = $content -replace 'logger\.Info\("([^"]+)"[^)]*\)', 'log.Println("[INFO] $1")'
        $content = $content -replace 'logger\.Warn\("([^"]+)"[^)]*\)', 'log.Println("[WARN] $1")'
        $content = $content -replace 'logger\.Error\("([^"]+)"[^)]*\)', 'log.Println("[ERROR] $1")'
        $content = $content -replace 'logger\.Debug\("([^"]+)"[^)]*\)', '// Debug: $1'
        
        $content | Set-Content $file -Encoding UTF8
        Write-Host "  ✓ Updated: $(Split-Path $file -Leaf)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Not found: $(Split-Path $file -Leaf)" -ForegroundColor Red
    }
}

Write-Host "`nRemoving logrus from $($logrusFiles.Count) files..." -ForegroundColor Yellow

foreach ($file in $logrusFiles) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        
        # Replace logrus import with log
        $content = $content -replace '"github\.com/sirupsen/logrus"', '"log"'
        
        # Replace all logrus calls
        $content = $content -replace 'logrus\.Info\("([^"]+)"[^)]*\)', 'log.Println("[INFO] $1")'
        $content = $content -replace 'logrus\.Infof\("([^"]+)"[^)]*\)', 'log.Printf("[INFO] $1\n", $2)'
        $content = $content -replace 'logrus\.Warn\("([^"]+)"[^)]*\)', 'log.Println("[WARN] $1")'
        $content = $content -replace 'logrus\.Warnf\("([^"]+)"[^)]*\)', 'log.Printf("[WARN] $1\n", $2)'
        $content = $content -replace 'logrus\.Error\("([^"]+)"[^)]*\)', 'log.Println("[ERROR] $1")'
        $content = $content -replace 'logrus\.Errorf\("([^"]+)"[^)]*\)', 'log.Printf("[ERROR] $1\n", $2)'
        
        $content | Set-Content $file -Encoding UTF8
        Write-Host "  ✓ Updated: $(Split-Path $file -Leaf)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Not found: $(Split-Path $file -Leaf)" -ForegroundColor Red
    }
}

Write-Host "`nRemoving uuid dependency..." -ForegroundColor Yellow

$uuidFile = "$rootPath\internal\ssrf\oob\monitor.go"
if (Test-Path $uuidFile) {
    $content = Get-Content $uuidFile -Raw
    
    # Replace uuid import
    $content = $content -replace '"github\.com/google/uuid"', '"crypto/rand"
	"encoding/hex"'
    
    # Replace uuid.New().String()
    $content = $content -replace 'uuid\.New\(\)\.String\(\)', 'generateUUID()'
    
    # Add generateUUID function after imports
    if ($content -notmatch 'func generateUUID') {
        $uuidFunc = @"

// generateUUID generates a simple UUID-like string using crypto/rand
func generateUUID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}
"@
        # Insert after the comment block
        $content = $content -replace '(// ={70}\n)', "`$1$uuidFunc`n"
    }
    
    $content | Set-Content $uuidFile -Encoding UTF8
    Write-Host "  ✓ Updated: monitor.go" -ForegroundColor Green
}

Write-Host "`nUpdating go.mod..." -ForegroundColor Yellow

$goModFile = "$rootPath\go.mod"
$content = @"
module github.com/kyle0527/aiva/services/scan/engines/go_engine

go 1.23.1

// No external dependencies - using standard library only
// Following aiva_common development standards
"@

$content | Set-Content $goModFile -Encoding UTF8
Write-Host "  ✓ Updated go.mod" -ForegroundColor Green

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Dependency cleanup complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. cd $rootPath"
Write-Host "  2. go mod tidy"
Write-Host "  3. go build ./..."
