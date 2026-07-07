# Réentraînement hebdomadaire du modèle de sentiment (Windows).
#
# Crée une tâche planifiée "SocialMetrics-Retrain" qui exécute
# scripts/retrain.py tous les lundis à 03h00.
#
# Usage (PowerShell en administrateur, depuis la racine du projet) :
#   .\scripts\setup_windows_task.ps1

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "scripts\retrain.py" `
    -WorkingDirectory $projectRoot

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 3:00AM

Register-ScheduledTask `
    -TaskName "SocialMetrics-Retrain" `
    -Action $action `
    -Trigger $trigger `
    -Description "Réentraînement hebdomadaire du modèle d'analyse de sentiments (SocialMetrics AI)" `
    -Force

Write-Host "Tâche planifiée 'SocialMetrics-Retrain' créée (tous les lundis à 03h00)."
