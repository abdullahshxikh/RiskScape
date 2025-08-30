#!/bin/bash

# RiskScape Deployment Script
# This script builds and prepares the application for deployment

set -e

echo "🚀 Starting RiskScape deployment build..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    echo "🔍 Checking dependencies..."

    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18+"
        exit 1
    fi

    if ! command -v rustc &> /dev/null; then
        print_error "Rust is not installed. Please install Rust 1.70+"
        exit 1
    fi

    if ! command -v wasm-pack &> /dev/null; then
        print_error "wasm-pack is not installed. Please install wasm-pack"
        print_warning "Run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
        exit 1
    fi

    print_status "All dependencies are installed"
}

# Clean previous builds
clean_build() {
    echo "🧹 Cleaning previous builds..."
    rm -rf web/.next web/out engine/pkg engine/target
    print_status "Clean completed"
}

# Build WebAssembly module
build_wasm() {
    echo "🔨 Building WebAssembly module..."
    cd engine
    wasm-pack build --target web --out-dir pkg --release
    cd ..
    print_status "WebAssembly build completed"
}

# Build Next.js application
build_nextjs() {
    echo "🔨 Building Next.js application..."
    cd web
    npm run build
    cd ..
    print_status "Next.js build completed"
}

# Copy WASM files to public directory
copy_wasm_files() {
    echo "📋 Copying WASM files to public directory..."
    mkdir -p web/public/wasm
    cp engine/pkg/* web/public/wasm/
    print_status "WASM files copied"
}

# Create deployment package
create_deployment_package() {
    echo "📦 Creating deployment package..."
    mkdir -p dist
    cp -r web/.next dist/
    cp -r web/public dist/
    cp web/package.json dist/
    print_status "Deployment package created in ./dist/"
}

# Main deployment process
main() {
    echo "🎯 RiskScape Deployment Script"
    echo "=============================="

    check_dependencies
    clean_build
    build_wasm
    copy_wasm_files
    build_nextjs
    create_deployment_package

    echo ""
    print_status "Deployment build completed successfully!"
    echo ""
    echo "📁 Deployment files are ready in the ./dist/ directory"
    echo "🚀 You can now deploy the contents of ./dist/ to your hosting platform"
    echo ""
    echo "For Vercel/Netlify deployment:"
    echo "  - Upload the ./web/ directory (not ./dist/)"
    echo "  - The build scripts will handle the rest automatically"
}

# Run main function
main "$@"
