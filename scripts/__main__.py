def show_menu():
    print("=== Vision Sorting System ===")
    print("1. Capture new images")
    print("2. Preprocess dataset")
    print("3. Train traditional ML model")
    print("4. Train PyTorch CNN model")
    print("5. Run real-time inference")

    choice = input("Choose an option (1–5): ").strip()

    if choice == "1":
        import capture_and_save
    elif choice == "2":
        import load_and_preprocess
    elif choice == "3":
        import train_basic_ml
    elif choice == "4":
        import train_cnn_pytorch
    elif choice == "5":
        import realtime_inference
    else:
        print("Invalid choice. Please run the container again.")

if __name__ == "__main__":
    show_menu()
