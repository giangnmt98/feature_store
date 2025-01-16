pipeline {
    agent any
    environment {
        CODE_DIRECTORY = 'featurestore'

    }
    options {
        timestamps()
        disableConcurrentBuilds()
    }
    stages {
        stage('Check Code') {
            steps {
                script {
                    // Check line count and changes
                    sh '''
                    echo "=== Checking lines of code in each file ==="
                    MAX_LINES=500
                    set +x
                    git ls-files | grep -Ev ".pylintrc|airflow.cfg|data" | while read -r file; do
                        line_count=$(wc -l < "$file")
                        if [ "$line_count" -gt "$MAX_LINES" ]; then
                            echo "Error: File $file has $line_count lines, which exceeds the threshold of $MAX_LINES lines."
                            exit 1
                        fi
                    done
                    echo "=== Checking lines of code changes ==="
                    MAX_CHANGE_LINES=200
                    git fetch origin main
                    LAST_MAIN_COMMIT=$(git rev-parse origin/main)
                    CURRENT_COMMIT=$(git rev-parse HEAD)
                    if [ "$LAST_MAIN_COMMIT" = "$CURRENT_COMMIT" ]; then
                        DIFF_RANGE="HEAD^1..HEAD"
                    else
                        DIFF_RANGE="origin/main..HEAD"
                    fi
                    CHANGES=$(git diff --numstat "$DIFF_RANGE" | awk '{added+=$1; deleted+=$2} END {print added+deleted}')
                    if [ -n "$CHANGES" ] && [ "$CHANGES" -gt "$MAX_CHANGE_LINES" ]; then
                        echo "Error: Too many changes: $CHANGES lines."
                        exit 1
                    else
                        echo "Number of changed lines: $CHANGES"
                    fi
                    '''
                }
            }
        }
        stage('Setup and Run Pipeline') {
            agent {
                docker {
                    image 'test'
                    args '-v /home/giang/.ssh:/home/docker/.ssh --gpus all'
                }
            }
            steps {
                script {
                    // Set up Python environment
                    sh '''
                    python3 -m pip install --user --cache-dir /opt/conda/pkgs -e .[dev]
                    '''

                    // Run linting
                    sh '''
                    echo "=== Running Linting Tools ==="
                    flake8 $CODE_DIRECTORY
                    mypy --show-traceback $CODE_DIRECTORY
                    '''

                    // Run tests
                    sh '''
                    echo "=== Running Tests ==="
                    python3 -m pytest -s --durations=0 --disable-warnings tests/
                    '''
                }
            }
        }
    }
    post {
        success {
            echo "Pipeline completed successfully."
        }
        failure {
            echo "Pipeline failed."
        }
    }
}
