import project_setup
from data_details import count_file_types

def main():
    data_directory = 'homework_three/Fish'
    project_setup.create_homework_three_structure()

    file_counts = count_file_types(data_directory)

    print("\nDataset File Types")
    print("------------------")

    for file_type, count in file_counts.items():
        print(f"{file_type}: {count}")


if __name__ == "__main__":
    main()