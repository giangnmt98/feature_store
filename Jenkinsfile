pipeline {
    agent any
    environment {
        CODE_DIRECTORY = 'featurestore'
        TELEGRAM_BOT_TOKEN = '7897102108:AAEm888B6NUD4zRvlNfmvSCzNC94955cevg' // Thay bằng token của bot Telegram
        TELEGRAM_CHAT_ID = '2032100419'    // Thay bằng chat ID(Phải start chat với bot trước)  hoặc nhóm
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
                    sh '''
                    echo "a"
                    '''
                    // Set up Python environment
                    //sh '''
                    //export PATH=$PATH:/home/docker/.local/bin
                    //python3 -m pip install --user --cache-dir /opt/conda/pkgs -e .[dev]
                    //'''
                    //
                    //// Run linting
                    //sh '''
                    //echo "=== Running Linting Tools ==="
                    //python3 -m flake8 $CODE_DIRECTORY
                    //python3 -m mypy --show-traceback $CODE_DIRECTORY
                    //'''

                    //// Run tests
                    //sh '''
                    //echo "=== Running Tests ==="
                    //python3 -m pytest -s --durations=0 --disable-warnings tests/
                    //'''
                }
            }
        }
    }
post {
    success {
        script {
            // Tạo nội dung tin nhắn với MarkdownV2
            def MESSAGE = "✅ *Jenkins Pipeline Success* ✅\n" +
                          "*Job*: ${env.JOB_NAME}\n" +
                          "*Build*: ${env.BUILD_NUMBER}\n" +
                          "*View details*: ${env.BUILD_URL}"

            // Gửi thông báo qua Telegram
            sh """
            curl -s -X POST https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage \
            -d chat_id=${TELEGRAM_CHAT_ID} \
            -d parse_mode=MarkdownV2 \
            -d text="${MESSAGE}"
            """
        }
    }
    failure {
        script {
            // Tạo nội dung tin nhắn với MarkdownV2
            def MESSAGE = "🚨 *Jenkins Pipeline Failed* 🚨\n" +
                          "*Job*: ${env.JOB_NAME}\n" +
                          "*Build*: ${env.BUILD_NUMBER}\n" +
                          "*View details*: ${env.BUILD_URL}"

            // Gửi thông báo qua Telegram
            sh """
            curl -s -X POST https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage \
            -d chat_id=${TELEGRAM_CHAT_ID} \
            -d parse_mode=MarkdownV2 \
            -d text="${MESSAGE}"
            """
        }
    }
}
}
