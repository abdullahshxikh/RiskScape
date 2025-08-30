# RiskScape

Interactive Multi-Factor Risk Simulator in the Browser

## Features

- **Real-time Auto-Simulation**: Generate synthetic financial data continuously
- **Multi-Factor Regression**: Analyze asset returns against multiple risk factors
- **Principal Component Analysis**: Decompose risk into principal components
- **Covariance Analysis**: Compute and visualize correlation matrices
- **Stress Testing**: Test portfolio under various market scenarios
- **WebAssembly Backend**: High-performance Rust computations in the browser

## Tech Stack

- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS
- **Backend**: Rust WebAssembly (wasm-pack)
- **Charts**: Plotly.js for interactive visualizations
- **State**: Zustand for state management
- **UI**: Radix UI components

## Development

### Prerequisites

- Node.js 18+
- Rust 1.70+
- wasm-pack
- pnpm or npm

### Setup

```bash
# Install dependencies
npm install

# Build WebAssembly module
npm run build:wasm

# Start development server
npm run dev
```

### Build for Production

```bash
# Build everything
npm run build

# Or build individual parts
npm run build:wasm  # Build Rust WebAssembly
npm run build:web   # Build Next.js application
```

## Deployment

### Vercel

1. Connect your GitHub repository to Vercel
2. Vercel will automatically detect the Next.js project
3. The `vercel.json` configuration handles:
   - Build command: `npm run build`
   - Output directory: `.next`
   - WASM file caching headers
   - Security headers

### Netlify

1. Connect your GitHub repository to Netlify
2. Netlify will use the `netlify.toml` configuration:
   - Build command: `npm run build`
   - Publish directory: `.next`
   - Function directory: `src/pages/api/`
   - WASM file caching headers

### Manual Deployment

```bash
# Build the application
npm run build

# The built files will be in web/.next/
# Upload the web/ directory to your hosting provider
```

## Project Structure

```
riskscape/
├── engine/                 # Rust WebAssembly backend
│   ├── src/
│   │   ├── lib.rs         # Main WASM bindings
│   │   ├── model.rs       # Risk modeling functions
│   │   └── types.rs       # Type definitions
│   ├── Cargo.toml         # Rust dependencies
│   └── pkg/               # Compiled WASM files
├── web/                   # Next.js frontend
│   ├── src/
│   │   ├── app/           # Next.js app router
│   │   ├── components/    # React components
│   │   └── store/         # Zustand state management
│   ├── public/            # Static assets
│   └── package.json       # Node dependencies
├── .gitignore            # Git ignore rules
├── vercel.json           # Vercel configuration
├── netlify.toml          # Netlify configuration
└── package.json          # Root package configuration
```

## Environment Variables

Create a `.env.local` file for local development:

```bash
# Add any environment variables here
NODE_ENV=development
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details