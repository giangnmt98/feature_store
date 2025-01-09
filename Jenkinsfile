pipeline {
    agent any
    environment {
        CODE_DIRECTORY = 'featurestore'
        SSH_KEY = credentials('feathr_deploy_key')

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
                    MAX_LINES=700
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
                    args '-v /var/jenkins_home/.ssh:/dockeruser/.ssh --gpus all'
                }
            }
            steps {
                script {
                    // Set up Python environment
                    sh '''
                    echo "=== Setting up Python environment ==="
                    # Tạo thư mục nếu chưa tồn tại
                    mkdir -p /home/dockeruser/.ssh
                    chmod 700 /home/dockeruser/.ssh

                    # Thêm Host Git vào known_hosts
                    ssh-keyscan -H github-test-feathr-deploy >> /home/dockeruser/.ssh/known_hosts
                    chmod 600 /home/dockeruser/.ssh/known_hosts
                    # Thêm Host Git vào known_hosts
                    ssh-keyscan -H github-test-feathr-deploy >> ~/.ssh/known_hosts

                    # Sử dụng SSH key ed25519 để cài đặt package từ private Git repository
                    GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes" \
                    python3 -m pip install --cache-dir /opt/conda/pkgs -e .[dev]
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
