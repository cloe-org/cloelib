# 🏗️ Code Structure: Your Guide to cloelib's Architecture

Welcome to the **cloelib** architecture guide! 🚀

Think of **cloelib** as a cosmic LEGO set—modular, flexible, and designed to let you build amazing things. Whether you're computing gravitational lensing or analyzing galaxy clustering, our protocol-based architecture makes it easy to plug in new components and extend the library.

## 🎯 The Big Picture

**cloelib** follows a layered architecture that mirrors how we actually compute cosmological observables:

```
🌌 Background → 🌊 Perturbations → 🔭 Observables → 📊 Summary Statistics
```

Each layer builds on the previous one, creating a flexible pipeline from fundamental cosmology to final data products!

### Why This Design? 

**Modularity**: Want to swap CAMB for CLASS? Just plug in a different implementation!  
**Flexibility**: Need a custom tracer? Implement the protocol and you're done!  
**Reproducibility**: Clear interfaces mean everyone knows what's expected.  
**Fun**: Seriously, protocols make extending the library feel like solving a puzzle! 🧩

## 🧱 The Four Building Blocks

### 1. 🌌 [Background](background.md)

The foundation of everything! Background handles the cosmological stage—distances, Hubble parameter, matter densities. Think of it as setting up the universe before anything interesting happens.

**What it does**: Compute background quantities as functions of redshift  
**Key question**: "How far away is that galaxy?"  
**Interfaces with**: Nothing (it's the foundation!)  
**You'll love it if**: You're implementing a new Boltzmann solver or emulator

➡️ [Dive into Background](background.md)

### 2. 🌊 [Perturbations](perturbations.md)

Now things get interesting! Perturbations computes how structure forms and evolves—matter power spectra, growth factors, all the good stuff that makes galaxies cluster.

**What it does**: Calculate perturbation theory quantities  
**Key question**: "How lumpy is the universe at this scale and time?"  
**Interfaces with**: Background (it needs those distances!)  
**You'll love it if**: You're adding non-linear models or new structure formation codes

➡️ [Explore Perturbations](perturbations.md)

### 3. 🔭 [Observables](observables.md)

This is where we connect theory to what telescopes actually measure! Observables handles survey-specific calculations—selection functions, biases, window functions.

**What it does**: Compute survey-specific observables  
**Key question**: "What does my telescope see?"  
**Interfaces with**: Perturbations (for tracers) or Background (for spectro)  
**You'll love it if**: You're adding new types of measurements or survey configurations

➡️ [Check out Observables](observables.md)

### 4. 📊 [Summary Statistics](summary_statistics.md)

The grand finale! Summary Statistics produces the final data products you compare with observations—angular power spectra, correlation functions, multipoles.

**What it does**: Compute final statistical quantities  
**Key question**: "What numbers do I put in my likelihood?"  
**Interfaces with**: Observables (it needs those tracers!)  
**You'll love it if**: You're implementing new statistical estimators

➡️ [Discover Summary Statistics](summary_statistics.md)

## 🎮 Quick Start: The Workflow

Here's how everything flows together:

```python
# 1️⃣ Set up your cosmology
background = CAMBBackground(H0=67.5, Omega_b0=0.049, ...)

# 2️⃣ Add structure formation
perturbations = CAMBPerturbations(background=background, ...)

# 3️⃣ Define what you're observing
tracer = ShearTracer(perturbations=perturbations, dndz=..., z=..., ...)

# 4️⃣ Compute the statistics
two_point = AngularTwoPoint(tracer1=tracer, tracer2=tracer)
C_ell = two_point.compute_Cl(ells=...)  # 🎉 Done!
```

See? Each piece slots in naturally!

## 🛠️ For Contributors

Want to extend **cloelib**? You're in the right place! Each module page includes:

- ✅ **Protocol definitions**: What you need to implement
- ✅ **Step-by-step guides**: How to add your own implementations
- ✅ **Code examples**: Copy, paste, adapt!
- ✅ **Interface details**: How to connect with other codes
- ✅ **Testing tips**: Make sure everything works

The beauty of protocols is that you don't need to inherit from base classes or understand the entire codebase. Just implement the required methods, and you're golden! ✨

## 🎨 Design Philosophy

### Protocol-Based Design 🎯

We use Python protocols (PEP 544) instead of traditional inheritance. This means:

- **Duck typing with safety**: If it quacks like a Background, it is a Background!
- **Clear contracts**: Protocols explicitly document what's required
- **Flexibility**: No rigid class hierarchies to wrestle with

### Separation of Concerns 🎭

Each module has one job and does it well:

- **Background**: Pure cosmology (no structure formation)
- **Perturbations**: Structure growth (no survey details)
- **Observables**: Survey specifics (no final statistics)
- **Summary Statistics**: Final products (no cosmology details)

This separation makes the code easier to understand, test, and extend!

### JAX-First (But NumPy-Friendly) ⚡

Most implementations support both NumPy and JAX arrays:

- 🎓 **Automatic differentiation**: Compute gradients for free!
- 🚀 **GPU acceleration**: Scale to larger problems
- ⚡ **JIT compilation**: Blazing fast performance
- 🤝 **NumPy compatibility**: Use what you're comfortable with

## 📚 What's Next?

Ready to dive deeper? Pick a module that interests you:

- 🌌 [Background](background.md) - Start at the foundation
- 🌊 [Perturbations](perturbations.md) - Add structure formation
- 🔭 [Observables](observables.md) - Connect to surveys
- 📊 [Summary Statistics](summary_statistics.md) - Compute final products

Or jump to:

- 📖 [API Reference](../api.md) - Detailed technical documentation
- 🤝 [Contributing Guide](../contributing.md) - General contribution guidelines
- 💻 [Playground Examples](https://github.com/cloe-org/playground) - See it in action!

## 💬 Questions?

Stuck? Confused? Just curious? We're here to help!

- 💬 [GitHub Discussions](https://github.com/cloe-org/cloelib/discussions) - Ask questions, share ideas
- 🐛 [GitHub Issues](https://github.com/cloe-org/cloelib/issues) - Report bugs, request features  
- 👥 Tag `@cloe-maintainers` - Get help from the team

Remember: **cloelib** is meant to be fun (and scientifically robust)! 🎉 We're building tools to help us understand the universe—how cool is that? 🌌

Happy coding! 🚀
