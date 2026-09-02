try:
    from .run_simulation import main
except ImportError:
    from run_simulation import main


if __name__ == "__main__":
    main()
